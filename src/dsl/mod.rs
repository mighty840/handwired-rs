//! The runtime half of `#[handwired]`.
//!
//! The macro keeps the original function as the oracle, and emits a second
//! copy of its body in which every value that depends on an input is a
//! [`Val<T>`]: a wire in the circuit being built, typed like the Rust value it
//! stands for. Operators on `Val`s append operations to that circuit, so the
//! copy runs once, at build time, and leaves the whole program behind.
//! Values that don't depend on inputs, like loop counters and literals, stay
//! native Rust values, so `for` loops unroll and constants fold.
//!
//! The macro rewrites only what Rust can't overload: comparisons, `&&`/`||`,
//! shifts, indexing, assignments, `if`/`match` and casts, into calls of this
//! module.
//!
//! The contract: the circuit agrees with the function on every input for
//! which the function returns without panicking.

mod bits;
mod compiled;
mod control;
mod ops;
mod wide;

use std::cell::RefCell;
use std::fmt::Debug;
use std::marker::PhantomData;

use crate::circuit::Circuit;
use crate::wire::Wire;
use wide::Limbs;

pub use bits::{Shift, shl, shr};
pub use compiled::{
    Attrs, Compiled, EncodedRecords, HwType, Inputs, Params, Report, Rng, VerifyError, declare_scalar, iteration_limit,
    sample_variant,
};
pub use control::{
    Branch, CastTo, Cond, Conv, Elem, IfElse, Lift, Mux, Placeholder, StoreElem, Unify, cast, if_else, index,
    placeholder, store, unsupported,
};
/// `#[derive(HwType)]` on a struct with named fields; see the macros crate.
pub use handwired_rs_macros::HwType;
pub use ops::{Cmp, Logic, RunningSum, and, compare_native, eq, ge, gt, le, lt, ne, or, running_sums};

thread_local! {
    static CONTEXT: RefCell<Option<Circuit>> = const { RefCell::new(None) };
}

/// Run `f` on the circuit being built. Panics outside a build.
pub(crate) fn with_circuit<R>(f: impl FnOnce(&mut Circuit) -> R) -> R {
    CONTEXT.with(|context| {
        let mut context = context.borrow_mut();
        let circuit =
            context.as_mut().expect("handwired values can only be used while a #[handwired] circuit is built");
        f(circuit)
    })
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    Bool,
    Int,
    Float,
}

/// A Rust scalar type a circuit value can stand for: bool, an integer, f32,
/// or an enum whose variants carry no data, as its discriminant.
pub trait Scalar: Copy + PartialEq + Debug + 'static {
    const KIND: Kind;
    const NAME: &'static str;
    const MIN: f64;
    const MAX: f64;
    fn to_f64(self) -> f64;
    fn from_f64(value: f64) -> Self;
    /// A valid value to start from, e.g. 0 for numbers or an enum's first variant.
    fn default_value() -> Self {
        Self::from_f64(0.0)
    }
    /// The value as an exact integer, for integers beyond 2^53 too.
    fn to_i128(self) -> i128 {
        self.to_f64() as i128
    }
}

/// An integer type whose values can index a table.
pub trait IntScalar: Scalar {}

/// A number type with arithmetic: the integers and f32.
pub trait Num: Scalar {}

/// Integer types wider than 2^24 can hold values f32 can't represent exactly.
/// The type's width in bits, e.g. 8 for u8 and i8.
pub(crate) fn width<T: Scalar>() -> usize {
    let magnitude = (T::MAX + 1.0).log2() as usize;
    if T::MIN < 0.0 { magnitude + 1 } else { magnitude }
}

pub(crate) fn is_wide<T: Scalar>() -> bool {
    T::KIND == Kind::Int && T::MAX - T::MIN > 16_777_216.0
}

impl Scalar for bool {
    const KIND: Kind = Kind::Bool;
    const NAME: &'static str = "bool";
    const MIN: f64 = 0.0;
    const MAX: f64 = 1.0;
    fn to_f64(self) -> f64 {
        if self { 1.0 } else { 0.0 }
    }
    fn from_f64(value: f64) -> bool {
        assert!(value == 0.0 || value == 1.0, "a bool wire holds {value}, not 0 or 1");
        value == 1.0
    }
}

impl Scalar for f32 {
    const KIND: Kind = Kind::Float;
    const NAME: &'static str = "f32";
    const MIN: f64 = f32::MIN as f64;
    const MAX: f64 = f32::MAX as f64;
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(value: f64) -> f32 {
        value as f32
    }
}
impl Num for f32 {}

macro_rules! integer_scalars {
    ($($int:ident),*) => {$(
        impl Scalar for $int {
            const KIND: Kind = Kind::Int;
            const NAME: &'static str = stringify!($int);
            const MIN: f64 = $int::MIN as f64;
            const MAX: f64 = $int::MAX as f64;
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn from_f64(value: f64) -> $int {
                value as $int
            }
            fn to_i128(self) -> i128 {
                self as i128
            }
        }
        impl IntScalar for $int {}
        impl Num for $int {}
    )*};
}
integer_scalars!(u8, u16, u32, u64, usize, i8, i16, i32, i64);

/// A circuit value standing for a Rust value of type `T`: one wire, or for
/// an integer type wider than 24 bits whose value may leave ±2^24, its bytes,
/// one wire per 8 bits of its two's-complement pattern. Bytes whose value
/// fits within ±2^24 again, e.g. after a mask or a shift, come with the one
/// wire too; whichever form the circuit doesn't read is removed when it
/// compiles.
pub struct Val<T> {
    narrow: Option<Wire>,
    pub(crate) wide: Option<Limbs>,
    /// The value came from bytes: one-wire arithmetic on it that may leave ±2^24 goes back to bytes, as the bytes
    /// would have, instead of failing the build like arithmetic on a loose declared range.
    pub(crate) spill: bool,
    marker: PhantomData<T>,
}

impl<T> Clone for Val<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Val<T> {}

impl<T> Debug for Val<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.narrow, &self.wide) {
            (Some(wire), None) => write!(formatter, "Val({})", wire.index()),
            (narrow, Some(limbs)) => {
                let indices: Vec<usize> = limbs.wires().iter().map(|wire| wire.index()).collect();
                match narrow {
                    Some(wire) => write!(formatter, "Val({}, bytes {indices:?})", wire.index()),
                    None => write!(formatter, "Val(bytes {indices:?})"),
                }
            }
            (None, None) => unreachable!("a value has a wire or bytes"),
        }
    }
}

impl<T> Val<T> {
    /// The value's wire. Panics for a wide integer, which is several wires.
    #[track_caller]
    pub fn wire(&self) -> Wire {
        let Some(wire) = self.narrow else {
            panic!(
                "{}: this value may leave ±2^24, so it is held as bytes, and this operation needs it as one wire; \
                 narrow it first, e.g. with `as u8` or a mask",
                std::panic::Location::caller()
            );
        };
        wire
    }

    /// Whether the value is held only as bytes, without one wire.
    pub fn is_wide(&self) -> bool {
        self.narrow.is_none()
    }

    /// Whether the value has bytes, with or without one wire.
    pub(crate) fn has_bytes(&self) -> bool {
        self.wide.is_some()
    }

    /// The value, spilling into bytes like `spill` says.
    pub(crate) fn spilling(mut self, spill: bool) -> Val<T> {
        self.spill = spill;
        self
    }
}

impl<T: Scalar> Val<T> {
    pub fn from_wire(wire: Wire) -> Val<T> {
        Val { narrow: Some(wire), wide: None, spill: false, marker: PhantomData }
    }

    /// Bytes, with one wire as well where their value fits within ±2^24.
    pub(crate) fn from_limbs(limbs: Limbs) -> Val<T> {
        Val { narrow: wide::narrow_form::<T>(&limbs), wide: Some(limbs), spill: true, marker: PhantomData }
    }

    /// One wire and the bytes of the same value.
    pub(crate) fn both(wire: Wire, limbs: Limbs) -> Val<T> {
        Val { narrow: Some(wire), wide: Some(limbs), spill: true, marker: PhantomData }
    }

    /// A wire that always holds `value`, or bytes for an integer beyond ±2^24.
    pub fn constant(value: T) -> Val<T> {
        if T::KIND == Kind::Int && value.to_i128().unsigned_abs() >= 1 << 24 {
            return Val::from_limbs(wide::constant_limbs::<T>(value.to_i128()));
        }
        with_circuit(|circuit| {
            let wire = circuit.constant(value.to_f64());
            if T::KIND != Kind::Float {
                circuit.assume_grid_allowing_readers(wire, 1.0);
            }
            Val::from_wire(wire)
        })
    }
}

/// The value of a bit that is a constant wire, e.g. a flag that no
/// input-dependent branch has set yet, so selects on it fold away.
pub(crate) fn known_bit(bit: Val<bool>) -> Option<bool> {
    with_circuit(|circuit| circuit.constant_value(bit.wire())).map(|value| value as f32 != 0.0)
}

/// Narrow a wire to what its type promises about every non-panicking value,
/// e.g. 0..1 on a grid of 1 for a bool, so later comparisons stay cheap.
pub(crate) fn settle<T: Scalar>(circuit: &mut Circuit, wire: Wire, min: f64, max: f64) -> Val<T> {
    let range = circuit.range(wire);
    let (min, max) = (range.min.max(min), range.max.min(max));
    if min <= max {
        circuit.assume_range_allowing_readers(wire, min, max);
    }
    if T::KIND != Kind::Float && wire.index() >= circuit.source_size() {
        circuit.assume_grid_allowing_readers(wire, 1.0);
    }
    Val::from_wire(wire)
}
