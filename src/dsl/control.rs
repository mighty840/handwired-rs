//! Conversions, selection, branching, indexing and casts.

use std::panic::Location;

use super::ops::integer_table;
use super::{IntScalar, Kind, Scalar, Val, eq, gt, is_wide, known_bit, settle, with_circuit};
use crate::circuit::OutOfRange;
use crate::emit::{Relation, emit_compare, emit_round};
use crate::wire::{Lin, Wire};

/// The build stops here: the macro puts this where a construct can't be
/// translated, e.g. a `return` inside a branch whose condition depends on
/// the inputs.
#[track_caller]
pub fn unsupported(what: &str) -> ! {
    panic!("{}: {what}", Location::caller())
}

// ---- Selection --------------------------------------------------------------------------------------------------------

/// A value a circuit can choose between: `bit ? on : off`, with both sides computed.
pub trait Mux: Copy {
    fn select(bit: Val<bool>, off: Self, on: Self) -> Self;

    /// `items[index]` at an index that depends on the inputs. Out of bounds
    /// reads the first item; the oracle would have panicked there.
    fn read_dynamic<I: IntScalar>(items: &[Self], index: Val<I>) -> Self {
        let mut chosen = items[0];
        for (position, &item) in items.iter().enumerate().skip(1) {
            chosen = Self::select(eq_position(index, position), chosen, item);
        }
        chosen
    }
}

fn eq_position<I: IntScalar>(index: Val<I>, position: usize) -> Val<bool> {
    eq(index, I::from_f64(position as f64))
}

impl<T: Scalar> Mux for Val<T> {
    fn select(bit: Val<bool>, off: Val<T>, on: Val<T>) -> Val<T> {
        if off.wire == on.wire {
            return off;
        }
        if let Some(value) = known_bit(bit) {
            return if value { on } else { off };
        }
        with_circuit(|circuit| {
            let (off_range, on_range) = (circuit.range(off.wire), circuit.range(on.wire));
            let wire = circuit.select(bit.wire, off.wire, on.wire);
            // Either side is a T wherever the function doesn't panic, so the result stays in T's range too; this
            // keeps loop-carried integers, e.g. gcd's `a -= b` over 254 unrolled passes, from widening every pass.
            let (low, high) = (off_range.min.min(on_range.min), off_range.max.max(on_range.max));
            match T::KIND {
                Kind::Float => settle(circuit, wire, low, high),
                _ => settle(circuit, wire, low.max(T::MIN), high.min(T::MAX)),
            }
        })
    }

    fn read_dynamic<I: IntScalar>(items: &[Val<T>], index: Val<I>) -> Val<T> {
        let candidates: Vec<Wire> = items.iter().map(|item| item.wire).collect();
        with_circuit(|circuit| {
            let wire = circuit.index_read(index.wire, &candidates, OutOfRange::Clamp);
            let range = circuit.range(wire);
            settle(circuit, wire, range.min, range.max)
        })
    }
}

impl Mux for () {
    fn select(_: Val<bool>, _: (), _: ()) {}
}

impl<M: Mux, const N: usize> Mux for [M; N] {
    fn select(bit: Val<bool>, off: [M; N], on: [M; N]) -> [M; N] {
        std::array::from_fn(|position| M::select(bit, off[position], on[position]))
    }
}

/// A stand-in value for a branch that ends in `return`, `break` or
/// `continue`: the branch's value is never used on the paths that take it,
/// but both branches of an input-dependent `if` are computed.
pub trait Placeholder: Copy {
    fn placeholder() -> Self;
}
impl<T: Scalar> Placeholder for Val<T> {
    fn placeholder() -> Val<T> {
        Val::constant(T::default_value())
    }
}
impl Placeholder for () {
    fn placeholder() {}
}
impl<P: Placeholder, const N: usize> Placeholder for [P; N] {
    fn placeholder() -> [P; N] {
        [P::placeholder(); N]
    }
}

pub fn placeholder<P: Placeholder>() -> P {
    P::placeholder()
}

// ---- Conversions ------------------------------------------------------------------------------------------------------

/// Turn a value into the circuit form of a known type, e.g. a literal `5`
/// assigned to a `u8` variable that holds a circuit value.
pub trait Conv<T> {
    fn conv(self) -> T;
}

impl<T: Scalar> Conv<Val<T>> for Val<T> {
    fn conv(self) -> Val<T> {
        self
    }
}
impl Conv<()> for () {
    fn conv(self) {}
}
impl<A: Conv<B>, B, const N: usize> Conv<[B; N]> for [A; N] {
    fn conv(self) -> [B; N] {
        self.map(Conv::conv)
    }
}

/// The circuit form of a value whose type is not known yet, for `let mut`
/// bindings that later take circuit values.
pub trait Lift {
    type Out;
    fn lift(self) -> Self::Out;
}

impl<T: Scalar> Lift for Val<T> {
    type Out = Val<T>;
    fn lift(self) -> Val<T> {
        self
    }
}
impl Lift for () {
    type Out = ();
    fn lift(self) {}
}
impl<L: Lift, const N: usize> Lift for [L; N] {
    type Out = [L::Out; N];
    fn lift(self) -> [L::Out; N] {
        self.map(Lift::lift)
    }
}

/// The common type of an `if`'s two branch values: native when both are
/// native, a circuit value when either is.
pub trait Unify<B>: Sized {
    type Out;
    fn left(self) -> Self::Out;
    fn right(other: B) -> Self::Out;
}

impl<T: Scalar> Unify<Val<T>> for Val<T> {
    type Out = Val<T>;
    fn left(self) -> Val<T> {
        self
    }
    fn right(other: Val<T>) -> Val<T> {
        other
    }
}
impl<T: Scalar> Unify<T> for Val<T> {
    type Out = Val<T>;
    fn left(self) -> Val<T> {
        self
    }
    fn right(other: T) -> Val<T> {
        Val::constant(other)
    }
}
impl Unify<()> for () {
    type Out = ();
    fn left(self) {}
    fn right(_: ()) {}
}
impl<A: Unify<B>, B, const N: usize> Unify<[B; N]> for [A; N] {
    type Out = [A::Out; N];
    fn left(self) -> [A::Out; N] {
        self.map(A::left)
    }
    fn right(other: [B; N]) -> [A::Out; N] {
        other.map(A::right)
    }
}

macro_rules! native_conversions {
    ($($t:ident),*) => {$(
        impl Conv<Val<$t>> for $t {
            fn conv(self) -> Val<$t> {
                Val::constant(self)
            }
        }
        impl Lift for $t {
            type Out = Val<$t>;
            fn lift(self) -> Val<$t> {
                Val::constant(self)
            }
        }
        impl Unify<$t> for $t {
            type Out = $t;
            fn left(self) -> $t {
                self
            }
            fn right(other: $t) -> $t {
                other
            }
        }
        impl Unify<Val<$t>> for $t {
            type Out = Val<$t>;
            fn left(self) -> Val<$t> {
                Val::constant(self)
            }
            fn right(other: Val<$t>) -> Val<$t> {
                other
            }
        }
    )*};
}
native_conversions!(bool, u8, u16, u32, u64, usize, i8, i16, i32, i64, f32);

macro_rules! tuple_traits {
    ($(($($name:ident $other:ident $index:tt),+)),*) => {$(
        impl<$($name: Placeholder),+> Placeholder for ($($name,)+) {
            fn placeholder() -> Self {
                ($($name::placeholder(),)+)
            }
        }
        impl<$($name: Mux),+> Mux for ($($name,)+) {
            fn select(bit: Val<bool>, off: Self, on: Self) -> Self {
                ($($name::select(bit, off.$index, on.$index),)+)
            }
        }
        impl<$($name: Conv<$other>, $other),+> Conv<($($other,)+)> for ($($name,)+) {
            fn conv(self) -> ($($other,)+) {
                ($(self.$index.conv(),)+)
            }
        }
        impl<$($name: Lift),+> Lift for ($($name,)+) {
            type Out = ($($name::Out,)+);
            fn lift(self) -> Self::Out {
                ($(self.$index.lift(),)+)
            }
        }
        impl<$($name: Unify<$other>, $other),+> Unify<($($other,)+)> for ($($name,)+) {
            type Out = ($($name::Out,)+);
            fn left(self) -> Self::Out {
                ($(self.$index.left(),)+)
            }
            fn right(other: ($($other,)+)) -> Self::Out {
                ($($name::right(other.$index),)+)
            }
        }
        impl<$($name: Copy),+> Elem<usize> for ($($name,)+) {
            type Out = Self;
            fn index(items: &[Self], index: usize) -> Self {
                items[index]
            }
        }
        impl<$($name: Copy),+, I: IntScalar> Elem<Val<I>> for ($($name,)+)
        where
            Self: Lift,
            <Self as Lift>::Out: Mux,
        {
            type Out = <Self as Lift>::Out;
            fn index(items: &[Self], index: Val<I>) -> Self::Out {
                let lifted: Vec<Self::Out> = items.iter().map(|&item| item.lift()).collect();
                Mux::read_dynamic(&lifted, index)
            }
        }
    )*};
}
tuple_traits!(
    (A0 B0 0),
    (A0 B0 0, A1 B1 1),
    (A0 B0 0, A1 B1 1, A2 B2 2),
    (A0 B0 0, A1 B1 1, A2 B2 2, A3 B3 3),
    (A0 B0 0, A1 B1 1, A2 B2 2, A3 B3 3, A4 B4 4),
    (A0 B0 0, A1 B1 1, A2 B2 2, A3 B3 3, A4 B4 4, A5 B5 5),
    (A0 B0 0, A1 B1 1, A2 B2 2, A3 B3 3, A4 B4 4, A5 B5 5, A6 B6 6),
    (A0 B0 0, A1 B1 1, A2 B2 2, A3 B3 3, A4 B4 4, A5 B5 5, A6 B6 6, A7 B7 7),
    (A0 B0 0, A1 B1 1, A2 B2 2, A3 B3 3, A4 B4 4, A5 B5 5, A6 B6 6, A7 B7 7, A8 B8 8),
    (A0 B0 0, A1 B1 1, A2 B2 2, A3 B3 3, A4 B4 4, A5 B5 5, A6 B6 6, A7 B7 7, A8 B8 8, A9 B9 9),
    (A0 B0 0, A1 B1 1, A2 B2 2, A3 B3 3, A4 B4 4, A5 B5 5, A6 B6 6, A7 B7 7, A8 B8 8, A9 B9 9, A10 B10 10),
    (A0 B0 0, A1 B1 1, A2 B2 2, A3 B3 3, A4 B4 4, A5 B5 5, A6 B6 6, A7 B7 7, A8 B8 8, A9 B9 9, A10 B10 10, A11 B11 11)
);

// ---- Branching --------------------------------------------------------------------------------------------------------

/// How an `if` runs: a native condition takes one branch, like Rust; a
/// circuit condition runs both and selects.
pub enum Branch {
    Static(bool),
    Dynamic(Val<bool>),
}

pub trait Cond: Copy {
    fn branch(self) -> Branch;
}
impl Cond for bool {
    fn branch(self) -> Branch {
        Branch::Static(self)
    }
}
impl Cond for Val<bool> {
    fn branch(self) -> Branch {
        Branch::Dynamic(self)
    }
}

/// An `if` whose branches only compute a value.
pub trait IfElse<A, B> {
    type Out;
    fn if_else(self, then: impl FnOnce() -> A, otherwise: impl FnOnce() -> B) -> Self::Out;
}

impl<A: Unify<B>, B> IfElse<A, B> for bool {
    type Out = A::Out;
    fn if_else(self, then: impl FnOnce() -> A, otherwise: impl FnOnce() -> B) -> A::Out {
        if self { then().left() } else { A::right(otherwise()) }
    }
}

impl<A: Unify<B>, B> IfElse<A, B> for Val<bool>
where
    A::Out: Lift,
    <A::Out as Lift>::Out: Mux,
{
    type Out = <A::Out as Lift>::Out;
    fn if_else(self, then: impl FnOnce() -> A, otherwise: impl FnOnce() -> B) -> Self::Out {
        let on = then().left().lift();
        let off = A::right(otherwise()).lift();
        Mux::select(self, off, on)
    }
}

pub fn if_else<C: IfElse<A, B>, A, B>(condition: C, then: impl FnOnce() -> A, otherwise: impl FnOnce() -> B) -> C::Out {
    condition.if_else(then, otherwise)
}

// ---- Indexing ---------------------------------------------------------------------------------------------------------

/// `items[index]`: native for a native index, a table read for a circuit index.
pub trait Elem<I>: Sized {
    type Out;
    fn index(items: &[Self], index: I) -> Self::Out;
}

impl<T: Scalar> Elem<usize> for Val<T> {
    type Out = Val<T>;
    fn index(items: &[Val<T>], index: usize) -> Val<T> {
        items[index]
    }
}
impl<T: Scalar, I: IntScalar> Elem<Val<I>> for Val<T> {
    type Out = Val<T>;
    fn index(items: &[Val<T>], index: Val<I>) -> Val<T> {
        Mux::read_dynamic(items, index)
    }
}
impl<E: Copy, const N: usize> Elem<usize> for [E; N] {
    type Out = [E; N];
    fn index(items: &[[E; N]], index: usize) -> [E; N] {
        items[index]
    }
}
impl<E: Copy, const N: usize, I: IntScalar> Elem<Val<I>> for [E; N]
where
    [E; N]: Lift,
    <[E; N] as Lift>::Out: Mux,
{
    type Out = <[E; N] as Lift>::Out;
    fn index(items: &[[E; N]], index: Val<I>) -> Self::Out {
        let lifted: Vec<Self::Out> = items.iter().map(|&item| item.lift()).collect();
        Mux::read_dynamic(&lifted, index)
    }
}

macro_rules! native_elements {
    ($($t:ident),*) => {$(
        impl Elem<usize> for $t {
            type Out = $t;
            fn index(items: &[$t], index: usize) -> $t {
                items[index]
            }
        }
        impl<I: IntScalar> Elem<Val<I>> for $t {
            type Out = Val<$t>;
            fn index(items: &[$t], index: Val<I>) -> Val<$t> {
                let constants: Vec<Val<$t>> = items.iter().map(|&item| Val::constant(item)).collect();
                Mux::read_dynamic(&constants, index)
            }
        }
    )*};
}
native_elements!(bool, u8, u16, u32, u64, usize, i8, i16, i32, i64, f32);

pub fn index<E: Elem<I>, I, const N: usize>(items: &[E; N], index: I) -> E::Out {
    E::index(items, index)
}

/// `items[index] = value`.
pub trait StoreElem<I>: Sized {
    fn store(items: &mut [Self], index: I, value: Self);
}
impl<E> StoreElem<usize> for E {
    fn store(items: &mut [E], index: usize, value: E) {
        items[index] = value;
    }
}
impl<E: Mux, I: IntScalar> StoreElem<Val<I>> for E {
    fn store(items: &mut [E], index: Val<I>, value: E) {
        for (position, item) in items.iter_mut().enumerate() {
            *item = E::select(eq_position(index, position), *item, value);
        }
    }
}

pub fn store<E: StoreElem<I>, I, V: Conv<E>, const N: usize>(items: &mut [E; N], index: I, value: V) {
    E::store(items, index, value.conv());
}

// ---- Casts ------------------------------------------------------------------------------------------------------------

/// `value as T`.
pub trait CastTo<T> {
    type Out;
    fn cast_to(self) -> Self::Out;
}

#[track_caller]
pub fn cast<T, S: CastTo<T>>(value: S) -> S::Out {
    value.cast_to()
}

impl<S: Scalar, T: Scalar> CastTo<T> for Val<S> {
    type Out = Val<T>;
    #[track_caller]
    fn cast_to(self) -> Val<T> {
        cast_value::<S, T>(self, Location::caller())
    }
}

fn cast_value<S: Scalar, T: Scalar>(value: Val<S>, at: &'static Location<'static>) -> Val<T> {
    let same = Val::<T>::from_wire(value.wire);
    match (S::KIND, T::KIND) {
        (Kind::Bool, _) => same,
        (_, Kind::Bool) => unsupported("only bool casts to bool"),
        (Kind::Float, Kind::Float) => same,
        (Kind::Int, Kind::Float) => {
            let range = with_circuit(|circuit| circuit.range(value.wire));
            if range.min >= -16_777_216.0 && range.max <= 16_777_216.0 {
                same
            } else {
                panic!("{at}: casting integers beyond ±2^24 to f32 rounds, which isn't supported yet");
            }
        }
        (Kind::Int, Kind::Int) => {
            if S::MIN >= T::MIN && S::MAX <= T::MAX {
                return same;
            }
            let range = with_circuit(|circuit| circuit.range(value.wire));
            if range.min >= T::MIN && range.max <= T::MAX {
                return same;
            }
            if is_wide::<T>() {
                panic!("{at}: this cast to {} may wrap, which is supported for types up to 16 bits", T::NAME);
            }
            // `as` between integers keeps the low bits: the same value modulo the target's span.
            let span = T::MAX - T::MIN + 1.0;
            integer_table(Val::<T>::from_wire(value.wire), at, |wide| {
                let wrapped = (wide as f64 - T::MIN).rem_euclid(span) + T::MIN;
                wrapped as i64
            })
        }
        (Kind::Float, Kind::Int) => truncate::<T>(value.wire, at),
    }
}

/// `x as T` for an f32 x: truncate toward zero, then saturate to T's range.
fn truncate<T: Scalar>(value: Wire, at: &'static Location<'static>) -> Val<T> {
    let range = with_circuit(|circuit| circuit.range(value));
    if !(range.min > -4_194_304.0 && range.max < 4_194_304.0) {
        panic!("{at}: casting f32 to an integer needs a declared range within ±2^22, got {}..{}", range.min, range.max);
    }
    let truncated = with_circuit(|circuit| {
        let rounded = emit_round(circuit, value);
        // Round to nearest, then step back toward zero where rounding went away from it.
        let rounded_up = emit_compare(circuit, rounded, Relation::Greater, value);
        let rounded_down = emit_compare(circuit, rounded, Relation::Less, value);
        let zero = circuit.constant(0.0);
        let positive = emit_compare(circuit, value, Relation::AtLeast, zero);
        let too_high = circuit.multiply(rounded_up, positive);
        let negative = circuit.affine(1.0 - positive);
        let too_low = circuit.multiply(rounded_down, negative);
        let wire = circuit.affine(Lin::new(vec![(rounded, 1.0), (too_high, -1.0), (too_low, 1.0)], 0.0));
        circuit.assume_grid_allowing_readers(wire, 1.0);
        Val::<T>::from_wire(wire)
    });
    if range.min >= T::MIN && range.max <= T::MAX {
        return truncated;
    }
    let low = Val::<T>::constant(T::from_f64(T::MIN));
    let high = Val::<T>::constant(T::from_f64(T::MAX));
    let above = Mux::select(gt(truncated, high), truncated, high);
    Mux::select(gt(low, above), above, low)
}

macro_rules! native_casts {
    ($($source:ident),*) => {$(
        native_casts!(@to $source: u8, u16, u32, u64, usize, i8, i16, i32, i64, f32);
    )*};
    (@to $source:ident: $($target:ident),*) => {$(
        impl CastTo<$target> for $source {
            type Out = $target;
            fn cast_to(self) -> $target {
                self as $target
            }
        }
    )*};
}
native_casts!(u8, u16, u32, u64, usize, i8, i16, i32, i64, f32);

macro_rules! bool_casts {
    ($($target:ident),*) => {$(
        impl CastTo<$target> for bool {
            type Out = $target;
            fn cast_to(self) -> $target {
                self as $target
            }
        }
    )*};
}
bool_casts!(u8, u16, u32, u64, usize, i8, i16, i32, i64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;

    #[test]
    fn conversions_infer_from_context() {
        super::super::CONTEXT.with(|context| *context.borrow_mut() = Some(Circuit::new(0)));
        let value: Val<u8> = Conv::conv(5u8);
        let picked = if_else(true, || value, || 7);
        let _: Val<u8> = picked;
        let native = if_else(false, || 1, || 2);
        assert_eq!(native, 2);
        super::super::CONTEXT.with(|context| *context.borrow_mut() = None);
    }
}
