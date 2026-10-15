//! Arithmetic, comparisons and logic on circuit values.
//!
//! Integers lie on a grid of 1, so their sums are exact while they stay
//! within 2^24, and they use ordinary affine sums, which the builder merges.
//! f32 arithmetic must round exactly where Rust rounds, so every f32 `+`, `-`
//! and `*` is its own rounding-preserving operation, in written order.

use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Sub};
use std::panic::Location;

use super::{Kind, Num, Scalar, Val, is_wide, known_bit, settle, with_circuit};
use crate::circuit::{Circuit, OutOfRange};
use crate::emit::{Relation, emit_and, emit_compare, emit_or};
use crate::numeric::lowest_power_of_two;
use crate::wire::{Lin, Wire};

const EXACT_LIMIT: f64 = 16_777_216.0;

/// An integer result, checked to stay exact when its type is wider than 2^24.
fn number<T: Scalar>(circuit: &mut Circuit, wire: Wire, at: &Location) -> Val<T> {
    if is_wide::<T>() {
        let range = circuit.range(wire);
        if !(range.min > -EXACT_LIMIT && range.max < EXACT_LIMIT) {
            panic!(
                "{at}: this {} may reach {}..{}, beyond ±2^24 where f32 holds every integer; declare narrower \
                 #[range]s on the inputs",
                T::NAME,
                range.min,
                range.max
            );
        }
    }
    Val::from_wire(wire)
}

/// Append `form`: an ordinary sum for integers, a rounding-preserving one for f32.
#[track_caller]
fn sum<T: Scalar>(form: Lin) -> Val<T> {
    let at = Location::caller();
    with_circuit(|circuit| {
        let wire = if T::KIND == Kind::Float { circuit.round_affine(form) } else { circuit.affine(form) };
        number(circuit, wire, at)
    })
}

#[track_caller]
fn product<T: Scalar>(left: Wire, right: Wire) -> Val<T> {
    let at = Location::caller();
    with_circuit(|circuit| {
        let wire = circuit.multiply(left, right);
        number(circuit, wire, at)
    })
}

fn constant_wire<T: Scalar>(value: T) -> Wire {
    Val::constant(value).wire
}

// ---- Arithmetic -------------------------------------------------------------------------------------------------------

impl<T: Num> Add for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn add(self, right: Val<T>) -> Val<T> {
        sum(self.wire + right.wire)
    }
}
impl<T: Num> Add<T> for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn add(self, right: T) -> Val<T> {
        sum(self.wire + right.to_f64())
    }
}
impl<T: Num> Sub for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn sub(self, right: Val<T>) -> Val<T> {
        sum(self.wire - right.wire)
    }
}
impl<T: Num> Sub<T> for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn sub(self, right: T) -> Val<T> {
        sum(self.wire - right.to_f64())
    }
}
impl<T: Num> Mul for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn mul(self, right: Val<T>) -> Val<T> {
        product(self.wire, right.wire)
    }
}
impl<T: Num> Mul<T> for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn mul(self, right: T) -> Val<T> {
        sum(self.wire * right.to_f64())
    }
}
impl<T: Num> Neg for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn neg(self) -> Val<T> {
        sum(-self.wire)
    }
}
impl<T: Num> Div<T> for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn div(self, right: T) -> Val<T> {
        let at = Location::caller();
        if T::KIND == Kind::Float {
            let divisor = right.to_f64();
            if divisor != 0.0 && divisor.is_finite() && lowest_power_of_two(divisor) == divisor.abs() {
                return sum(self.wire * (1.0 / divisor));
            }
            panic!("{at}: f32 division is only supported by a power of two, got {divisor}");
        }
        integer_table(self, at, |value| {
            let divisor = right.to_f64() as i64;
            assert!(divisor != 0, "{at}: division by zero");
            value / divisor
        })
    }
}
impl<T: Num> Rem<T> for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn rem(self, right: T) -> Val<T> {
        let at = Location::caller();
        assert!(T::KIND == Kind::Int, "{at}: % is only supported on integers");
        integer_table(self, at, |value| {
            let divisor = right.to_f64() as i64;
            assert!(divisor != 0, "{at}: remainder by zero");
            value % divisor
        })
    }
}
impl<T: Num> Div for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn div(self, _: Val<T>) -> Val<T> {
        panic!("{}: division by a value that depends on the inputs isn't supported yet", Location::caller())
    }
}
impl<T: Num> Rem for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn rem(self, _: Val<T>) -> Val<T> {
        panic!("{}: remainder by a value that depends on the inputs isn't supported yet", Location::caller())
    }
}

/// Any integer function of one integer with few possible inputs, as one
/// table read: e.g. `x / 3` over x in 0..255 reads a table of 256 quotients.
pub(crate) fn integer_table<T: Scalar>(value: Val<T>, at: &Location, f: impl Fn(i64) -> i64) -> Val<T> {
    const TABLE_LIMIT: f64 = 65_536.0;
    let range = with_circuit(|circuit| circuit.range(value.wire));
    if !(range.is_finite() && range.max - range.min < TABLE_LIMIT) {
        panic!("{at}: this needs a table over {}..{}, more than 65,536 entries", range.min, range.max);
    }
    let (low, high) = (range.min.ceil() as i64, range.max.floor() as i64);
    let results: Vec<i64> = (low..=high).map(f).collect();
    with_circuit(|circuit| {
        let table: Vec<Wire> = results.iter().map(|&result| circuit.constant(result as f64)).collect();
        let read =
            circuit.index_read_affine(value.wire - low as f64, &table, OutOfRange::Clamp, OutOfRange::Clamp, None);
        circuit.assume_grid_allowing_readers(read, 1.0);
        number(circuit, read, at)
    })
}

macro_rules! native_left_arithmetic {
    ($($t:ident),*) => {$(
        impl Add<Val<$t>> for $t {
            type Output = Val<$t>;
            #[track_caller]
            fn add(self, right: Val<$t>) -> Val<$t> {
                // Addition is commutative in IEEE arithmetic too.
                sum(right.wire + self.to_f64())
            }
        }
        impl Sub<Val<$t>> for $t {
            type Output = Val<$t>;
            #[track_caller]
            fn sub(self, right: Val<$t>) -> Val<$t> {
                sum(self.to_f64() - right.wire)
            }
        }
        impl Mul<Val<$t>> for $t {
            type Output = Val<$t>;
            #[track_caller]
            fn mul(self, right: Val<$t>) -> Val<$t> {
                sum(right.wire * self.to_f64())
            }
        }
        impl Div<Val<$t>> for $t {
            type Output = Val<$t>;
            #[track_caller]
            fn div(self, right: Val<$t>) -> Val<$t> {
                let _ = right;
                panic!("{}: division by a value that depends on the inputs isn't supported yet", Location::caller())
            }
        }
    )*};
}
native_left_arithmetic!(u8, u16, u32, u64, usize, i8, i16, i32, i64, f32);

// ---- Methods ----------------------------------------------------------------------------------------------------------

impl<T: Num> Val<T> {
    fn select_by(bit: Val<bool>, off: Val<T>, on: Val<T>) -> Val<T> {
        <Val<T> as super::Mux>::select(bit, off, on)
    }

    #[track_caller]
    pub fn min(self, other: impl super::Conv<Val<T>>) -> Val<T> {
        let other = other.conv();
        // Ties return self, like Ord::min.
        Val::select_by(le(self, other), other, self)
    }

    #[track_caller]
    pub fn max(self, other: impl super::Conv<Val<T>>) -> Val<T> {
        let other = other.conv();
        // Ties return other, like Ord::max.
        Val::select_by(gt(self, other), other, self)
    }

    #[track_caller]
    pub fn clamp(self, low: impl super::Conv<Val<T>>, high: impl super::Conv<Val<T>>) -> Val<T> {
        self.max(low.conv()).min(high.conv())
    }

    #[track_caller]
    pub fn abs(self) -> Val<T> {
        let at = Location::caller();
        with_circuit(|circuit| {
            let positive = circuit.relu(self.wire);
            let negative = circuit.relu(-self.wire);
            let wire = if T::KIND == Kind::Float {
                circuit.round_affine(positive + negative)
            } else {
                circuit.affine(positive + negative)
            };
            number(circuit, wire, at)
        })
    }

    /// Integers only: `self + other` wrapped into the type's range.
    #[track_caller]
    pub fn wrapping_add(self, other: impl super::Conv<Val<T>>) -> Val<T> {
        self.wrap(self + other.conv())
    }

    #[track_caller]
    pub fn wrapping_sub(self, other: impl super::Conv<Val<T>>) -> Val<T> {
        self.wrap(self - other.conv())
    }

    #[track_caller]
    pub fn saturating_add(self, other: impl super::Conv<Val<T>>) -> Val<T> {
        self.saturate(self + other.conv())
    }

    #[track_caller]
    pub fn saturating_sub(self, other: impl super::Conv<Val<T>>) -> Val<T> {
        self.saturate(self - other.conv())
    }

    #[track_caller]
    fn wrap(self, value: Val<T>) -> Val<T> {
        let at = Location::caller();
        assert!(
            T::KIND == Kind::Int && !is_wide::<T>(),
            "{at}: wrapping arithmetic is supported on integers up to 16 bits"
        );
        // Subtract the span once per threshold the value can cross, e.g. 256 × [x >= 256] for a u8 sum.
        let span = T::MAX - T::MIN + 1.0;
        let range = with_circuit(|circuit| circuit.range(value.wire));
        let below = ((T::MIN - range.min) / span).ceil().max(0.0) as i64;
        let above = ((range.max - T::MAX) / span).ceil().max(0.0) as i64;
        let mut correction = Lin::new(vec![(value.wire, 1.0)], 0.0);
        for k in 1..=above {
            let threshold = Val::<T>::constant_f64(T::MIN + span * k as f64);
            let crossed = with_circuit(|circuit| emit_compare(circuit, value.wire, Relation::AtLeast, threshold));
            correction.terms.push((crossed, -span));
        }
        for k in 1..=below {
            let threshold = Val::<T>::constant_f64(T::MAX - span * k as f64);
            let crossed = with_circuit(|circuit| emit_compare(circuit, value.wire, Relation::AtMost, threshold));
            correction.terms.push((crossed, span));
        }
        let wrapped = sum::<T>(correction);
        with_circuit(|circuit| settle(circuit, wrapped.wire, T::MIN, T::MAX))
    }

    #[track_caller]
    fn saturate(self, value: Val<T>) -> Val<T> {
        let low = Val::<T>::from_wire(Val::<T>::constant_f64(T::MIN));
        let high = Val::<T>::from_wire(Val::<T>::constant_f64(T::MAX));
        value.max(low).min(high)
    }

    /// A constant wire from an f64, without passing through T, whose cast
    /// would saturate thresholds outside T's range, e.g. -1 for a u8.
    fn constant_f64(value: f64) -> Wire {
        with_circuit(|circuit| {
            let wire = circuit.constant(value);
            circuit.assume_grid_allowing_readers(wire, 1.0);
            wire
        })
    }
}

// ---- Comparisons ------------------------------------------------------------------------------------------------------

/// The six comparisons, on two circuit values, a value and a native, or two
/// natives, where they stay native.
pub trait Cmp<Right> {
    type Output;
    fn compare(self, relation: Relation, right: Right) -> Self::Output;
    fn equal(self, right: Right, equal: bool) -> Self::Output;
}

fn compare_wires(left: Wire, relation: Relation, right: Wire) -> Val<bool> {
    with_circuit(|circuit| Val::from_wire(emit_compare(circuit, left, relation, right)))
}

fn not_bit(bit: Val<bool>) -> Val<bool> {
    if let Some(value) = known_bit(bit) {
        return Val::constant(!value);
    }
    with_circuit(|circuit| Val::from_wire(circuit.affine(1.0 - bit.wire)))
}

fn equality(left: Wire, right: Wire, equal: bool) -> Val<bool> {
    let bit = compare_wires(left, Relation::Equal, right);
    if equal { bit } else { not_bit(bit) }
}

impl<T: Scalar> Cmp<Val<T>> for Val<T> {
    type Output = Val<bool>;
    fn compare(self, relation: Relation, right: Val<T>) -> Val<bool> {
        compare_wires(self.wire, relation, right.wire)
    }
    fn equal(self, right: Val<T>, equal: bool) -> Val<bool> {
        equality(self.wire, right.wire, equal)
    }
}

impl<T: Scalar> Cmp<T> for Val<T> {
    type Output = Val<bool>;
    fn compare(self, relation: Relation, right: T) -> Val<bool> {
        compare_wires(self.wire, relation, constant_wire(right))
    }
    fn equal(self, right: T, equal: bool) -> Val<bool> {
        equality(self.wire, constant_wire(right), equal)
    }
}

macro_rules! native_comparisons {
    ($($t:ident),*) => {$(
        impl Cmp<Val<$t>> for $t {
            type Output = Val<bool>;
            fn compare(self, relation: Relation, right: Val<$t>) -> Val<bool> {
                compare_wires(constant_wire(self), relation, right.wire)
            }
            fn equal(self, right: Val<$t>, equal: bool) -> Val<bool> {
                equality(constant_wire(self), right.wire, equal)
            }
        }
        impl Cmp<$t> for $t {
            type Output = bool;
            fn compare(self, relation: Relation, right: $t) -> bool {
                match relation {
                    Relation::Greater => self > right,
                    Relation::AtLeast => self >= right,
                    Relation::Less => self < right,
                    Relation::AtMost => self <= right,
                    Relation::Equal => self == right,
                }
            }
            fn equal(self, right: $t, equal: bool) -> bool {
                (self == right) == equal
            }
        }
    )*};
}
native_comparisons!(bool, u8, u16, u32, u64, usize, i8, i16, i32, i64, f32);

pub fn gt<A: Cmp<B>, B>(left: A, right: B) -> A::Output {
    left.compare(Relation::Greater, right)
}
pub fn ge<A: Cmp<B>, B>(left: A, right: B) -> A::Output {
    left.compare(Relation::AtLeast, right)
}
pub fn lt<A: Cmp<B>, B>(left: A, right: B) -> A::Output {
    left.compare(Relation::Less, right)
}
pub fn le<A: Cmp<B>, B>(left: A, right: B) -> A::Output {
    left.compare(Relation::AtMost, right)
}
pub fn eq<A: Cmp<B>, B>(left: A, right: B) -> A::Output {
    left.equal(right, true)
}
pub fn ne<A: Cmp<B>, B>(left: A, right: B) -> A::Output {
    left.equal(right, false)
}

// ---- Logic ------------------------------------------------------------------------------------------------------------

/// `&&` and `||`. A native left side short-circuits like Rust, so
/// `i > 0 && table[i - 1]` never evaluates `i - 1` at i = 0; a circuit left
/// side evaluates both.
pub trait Logic<Right> {
    type Output;
    fn and(self, right: impl FnOnce() -> Right) -> Self::Output;
    fn or(self, right: impl FnOnce() -> Right) -> Self::Output;
}

fn and_bits(left: Val<bool>, right: Val<bool>) -> Val<bool> {
    match (known_bit(left), known_bit(right)) {
        (Some(false), _) | (_, Some(false)) => return Val::constant(false),
        (Some(true), _) => return right,
        (_, Some(true)) => return left,
        _ => {}
    }
    with_circuit(|circuit| Val::from_wire(emit_and(circuit, &[left.wire, right.wire])))
}

fn or_bits(left: Val<bool>, right: Val<bool>) -> Val<bool> {
    match (known_bit(left), known_bit(right)) {
        (Some(true), _) | (_, Some(true)) => return Val::constant(true),
        (Some(false), _) => return right,
        (_, Some(false)) => return left,
        _ => {}
    }
    with_circuit(|circuit| {
        let wire = emit_or(circuit, &[left.wire, right.wire]);
        settle(circuit, wire, 0.0, 1.0)
    })
}

impl Logic<bool> for bool {
    type Output = bool;
    fn and(self, right: impl FnOnce() -> bool) -> bool {
        self && right()
    }
    fn or(self, right: impl FnOnce() -> bool) -> bool {
        self || right()
    }
}
impl Logic<Val<bool>> for bool {
    type Output = Val<bool>;
    fn and(self, right: impl FnOnce() -> Val<bool>) -> Val<bool> {
        if self { right() } else { Val::constant(false) }
    }
    fn or(self, right: impl FnOnce() -> Val<bool>) -> Val<bool> {
        if self { Val::constant(true) } else { right() }
    }
}
impl Logic<bool> for Val<bool> {
    type Output = Val<bool>;
    fn and(self, right: impl FnOnce() -> bool) -> Val<bool> {
        if right() { self } else { Val::constant(false) }
    }
    fn or(self, right: impl FnOnce() -> bool) -> Val<bool> {
        if right() { Val::constant(true) } else { self }
    }
}
impl Logic<Val<bool>> for Val<bool> {
    type Output = Val<bool>;
    fn and(self, right: impl FnOnce() -> Val<bool>) -> Val<bool> {
        and_bits(self, right())
    }
    fn or(self, right: impl FnOnce() -> Val<bool>) -> Val<bool> {
        or_bits(self, right())
    }
}

pub fn and<A: Logic<B>, B>(left: A, right: impl FnOnce() -> B) -> A::Output {
    left.and(right)
}
pub fn or<A: Logic<B>, B>(left: A, right: impl FnOnce() -> B) -> A::Output {
    left.or(right)
}

impl Not for Val<bool> {
    type Output = Val<bool>;
    fn not(self) -> Val<bool> {
        not_bit(self)
    }
}
impl BitAnd for Val<bool> {
    type Output = Val<bool>;
    fn bitand(self, right: Val<bool>) -> Val<bool> {
        and_bits(self, right)
    }
}
impl BitOr for Val<bool> {
    type Output = Val<bool>;
    fn bitor(self, right: Val<bool>) -> Val<bool> {
        or_bits(self, right)
    }
}
impl BitXor for Val<bool> {
    type Output = Val<bool>;
    fn bitxor(self, right: Val<bool>) -> Val<bool> {
        equality(self.wire, right.wire, false)
    }
}
impl BitAnd<bool> for Val<bool> {
    type Output = Val<bool>;
    fn bitand(self, right: bool) -> Val<bool> {
        if right { self } else { Val::constant(false) }
    }
}
impl BitOr<bool> for Val<bool> {
    type Output = Val<bool>;
    fn bitor(self, right: bool) -> Val<bool> {
        if right { Val::constant(true) } else { self }
    }
}
impl BitXor<bool> for Val<bool> {
    type Output = Val<bool>;
    fn bitxor(self, right: bool) -> Val<bool> {
        if right { not_bit(self) } else { self }
    }
}
impl BitAnd<Val<bool>> for bool {
    type Output = Val<bool>;
    fn bitand(self, right: Val<bool>) -> Val<bool> {
        right & self
    }
}
impl BitOr<Val<bool>> for bool {
    type Output = Val<bool>;
    fn bitor(self, right: Val<bool>) -> Val<bool> {
        right | self
    }
}
impl BitXor<Val<bool>> for bool {
    type Output = Val<bool>;
    fn bitxor(self, right: Val<bool>) -> Val<bool> {
        right ^ self
    }
}
