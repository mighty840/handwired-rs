//! Arithmetic, comparisons and logic on circuit values.
//!
//! Integers lie on a grid of 1, so their sums are exact while they stay
//! within 2^24, and they use ordinary affine sums, which the builder merges.
//! f32 arithmetic must round exactly where Rust rounds, so every f32 `+`, `-`
//! and `*` is its own rounding-preserving operation, in written order.

use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Sub};
use std::panic::Location;

use super::wide::{self, Limbs};
use super::{Kind, Num, Scalar, Val, is_wide, known_bit, settle, with_circuit};
use crate::circuit::{Circuit, OutOfRange};
use crate::emit::{Relation, emit_and, emit_compare, emit_or};
use crate::numeric::lowest_power_of_two;
use crate::wire::{Lin, Wire};

const EXACT_LIMIT: f64 = 16_777_216.0;

/// An integer result, checked to stay exact when its type is wider than 2^24.
pub(super) fn number<T: Scalar>(circuit: &mut Circuit, wire: Wire, at: &Location) -> Val<T> {
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

// ---- Arithmetic -------------------------------------------------------------------------------------------------------

/// Both sides as bytes, when either is already wide: arithmetic on them wraps, see [`wide`](super::wide).
pub(super) fn wide_pair<T: Scalar>(left: Val<T>, right: Val<T>) -> Option<(Limbs, Limbs)> {
    (left.is_wide() || right.is_wide()).then(|| (wide::limbs_as::<T, T>(left), wide::limbs_as::<T, T>(right)))
}

/// A native operand as a circuit value: bytes beyond ±2^24, where `wide_pair` takes over.
fn beyond<T: Scalar>(value: T) -> bool {
    T::KIND == Kind::Int && value.to_i128().unsigned_abs() >= 1 << 24
}

impl<T: Num> Add for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn add(self, right: Val<T>) -> Val<T> {
        if let Some((a, b)) = wide_pair(self, right) {
            return wide::add(&a, &b);
        }
        sum(self.wire() + right.wire())
    }
}
impl<T: Num> Add<T> for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn add(self, right: T) -> Val<T> {
        if self.is_wide() || beyond(right) {
            return self + Val::constant(right);
        }
        sum(self.wire() + right.to_f64())
    }
}
impl<T: Num> Sub for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn sub(self, right: Val<T>) -> Val<T> {
        if let Some((a, b)) = wide_pair(self, right) {
            return wide::sub(&a, &b);
        }
        sum(self.wire() - right.wire())
    }
}
impl<T: Num> Sub<T> for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn sub(self, right: T) -> Val<T> {
        if self.is_wide() || beyond(right) {
            return self - Val::constant(right);
        }
        sum(self.wire() - right.to_f64())
    }
}
impl<T: Num> Mul for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn mul(self, right: Val<T>) -> Val<T> {
        if let Some((a, b)) = wide_pair(self, right) {
            return wide::mul(&a, &b);
        }
        product(self.wire(), right.wire())
    }
}
impl<T: Num> Mul<T> for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn mul(self, right: T) -> Val<T> {
        if self.is_wide() || beyond(right) {
            return self * Val::constant(right);
        }
        sum(self.wire() * right.to_f64())
    }
}
impl<T: Num> Neg for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn neg(self) -> Val<T> {
        if self.is_wide() {
            return Val::constant(T::from_f64(0.0)) - self;
        }
        sum(-self.wire())
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
                return sum(self.wire() * (1.0 / divisor));
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
    let range = with_circuit(|circuit| circuit.range(value.wire()));
    if !(range.is_finite() && range.max - range.min < TABLE_LIMIT) {
        panic!("{at}: this needs a table over {}..{}, more than 65,536 entries", range.min, range.max);
    }
    let (low, high) = (range.min.ceil() as i64, range.max.floor() as i64);
    let results: Vec<i64> = (low..=high).map(f).collect();
    with_circuit(|circuit| {
        let table: Vec<Wire> = results.iter().map(|&result| circuit.constant(result as f64)).collect();
        let read =
            circuit.index_read_affine(value.wire() - low as f64, &table, OutOfRange::Clamp, OutOfRange::Clamp, None);
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
                sum(right.wire() + self.to_f64())
            }
        }
        impl Sub<Val<$t>> for $t {
            type Output = Val<$t>;
            #[track_caller]
            fn sub(self, right: Val<$t>) -> Val<$t> {
                sum(self.to_f64() - right.wire())
            }
        }
        impl Mul<Val<$t>> for $t {
            type Output = Val<$t>;
            #[track_caller]
            fn mul(self, right: Val<$t>) -> Val<$t> {
                sum(right.wire() * self.to_f64())
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
            let positive = circuit.relu(self.wire());
            let negative = circuit.relu(-self.wire());
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
        let other = other.conv();
        match self.wrapping_in_bytes(other, |left, right| left.min + right.min, |left, right| left.max + right.max) {
            Some((a, b)) => wide::add(&a, &b),
            None if is_wide::<T>() => self + other,
            None => self.wrap(self + other),
        }
    }

    #[track_caller]
    pub fn wrapping_sub(self, other: impl super::Conv<Val<T>>) -> Val<T> {
        let other = other.conv();
        match self.wrapping_in_bytes(other, |left, right| left.min - right.max, |left, right| left.max - right.min) {
            Some((a, b)) => wide::sub(&a, &b),
            None if is_wide::<T>() => self - other,
            None => self.wrap(self - other),
        }
    }

    /// Integers only: `self * other` wrapped into the type's range. Products that may leave ±2^24 are computed
    /// in bytes, so they stay exact.
    #[track_caller]
    pub fn wrapping_mul(self, other: impl super::Conv<Val<T>>) -> Val<T> {
        let other = other.conv();
        let corners = |left: crate::numeric::Range, right: crate::numeric::Range| {
            [left.min * right.min, left.min * right.max, left.max * right.min, left.max * right.max]
        };
        let low = |left, right| corners(left, right).into_iter().fold(f64::INFINITY, f64::min);
        let high = |left, right| corners(left, right).into_iter().fold(f64::NEG_INFINITY, f64::max);
        match self.wrapping_in_bytes(other, low, high) {
            Some((a, b)) => {
                let product: Val<T> = wide::mul(&a, &b);
                if is_wide::<T>() { product } else { wide::narrow(&product.wide.unwrap()) }
            }
            None if is_wide::<T>() => self * other,
            None => self.wrap(self * other),
        }
    }

    /// The operands as bytes when a wrapping operation needs them: when either is wide, or when the result,
    /// bounded by `low` and `high` over the operands' ranges, may leave the type or ±2^24. `None` where the
    /// one-wire result already lies in the type's range, or wraps within ±2^24 for types up to 16 bits.
    fn wrapping_in_bytes(
        self,
        other: Val<T>,
        low: impl Fn(crate::numeric::Range, crate::numeric::Range) -> f64,
        high: impl Fn(crate::numeric::Range, crate::numeric::Range) -> f64,
    ) -> Option<(Limbs, Limbs)> {
        assert!(T::KIND == Kind::Int, "wrapping arithmetic is for integers");
        if let Some(pair) = wide_pair(self, other) {
            return Some(pair);
        }
        let (left, right) = with_circuit(|circuit| (circuit.range(self.wire()), circuit.range(other.wire())));
        let (low, high) = (low(left, right), high(left, right));
        let fits_type = low >= T::MIN && high <= T::MAX;
        let fits_wire = low > -16_777_216.0 && high < 16_777_216.0;
        if fits_type && fits_wire || !is_wide::<T>() && fits_wire {
            return None;
        }
        Some((wide::limbs_as::<T, T>(self), wide::limbs_as::<T, T>(other)))
    }

    #[track_caller]
    pub fn saturating_add(self, other: impl super::Conv<Val<T>>) -> Val<T> {
        let other = other.conv();
        Val::not_wide_for("saturating_add", self, other);
        self.saturate(self + other)
    }

    #[track_caller]
    pub fn saturating_sub(self, other: impl super::Conv<Val<T>>) -> Val<T> {
        let other = other.conv();
        Val::not_wide_for("saturating_sub", self, other);
        self.saturate(self - other)
    }

    /// Wide sums wrap, so clamping them afterwards would be wrong: refuse instead of building that.
    #[track_caller]
    fn not_wide_for(name: &str, left: Val<T>, right: Val<T>) {
        if left.is_wide() || right.is_wide() {
            panic!(
                "{}: {name} on a value beyond ±2^24 isn't supported yet; use wrapping_* or compare first",
                Location::caller()
            );
        }
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
        let range = with_circuit(|circuit| circuit.range(value.wire()));
        let below = ((T::MIN - range.min) / span).ceil().max(0.0) as i64;
        let above = ((range.max - T::MAX) / span).ceil().max(0.0) as i64;
        let mut correction = Lin::new(vec![(value.wire(), 1.0)], 0.0);
        for k in 1..=above {
            let threshold = Val::<T>::constant_f64(T::MIN + span * k as f64);
            let crossed = with_circuit(|circuit| emit_compare(circuit, value.wire(), Relation::AtLeast, threshold));
            correction.terms.push((crossed, -span));
        }
        for k in 1..=below {
            let threshold = Val::<T>::constant_f64(T::MAX - span * k as f64);
            let crossed = with_circuit(|circuit| emit_compare(circuit, value.wire(), Relation::AtMost, threshold));
            correction.terms.push((crossed, span));
        }
        let wrapped = sum::<T>(correction);
        with_circuit(|circuit| settle(circuit, wrapped.wire(), T::MIN, T::MAX))
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
    with_circuit(|circuit| Val::from_wire(circuit.affine(1.0 - bit.wire())))
}

pub(super) fn equality(left: Wire, right: Wire, equal: bool) -> Val<bool> {
    let bit = compare_wires(left, Relation::Equal, right);
    if equal { bit } else { not_bit(bit) }
}

impl<T: Scalar> Cmp<Val<T>> for Val<T> {
    type Output = Val<bool>;
    fn compare(self, relation: Relation, right: Val<T>) -> Val<bool> {
        if let Some((a, b)) = wide_pair(self, right) {
            return match relation {
                Relation::AtLeast => wide::at_least::<T>(&a, &b),
                Relation::AtMost => wide::at_least::<T>(&b, &a),
                Relation::Less => not_bit(wide::at_least::<T>(&a, &b)),
                Relation::Greater => not_bit(wide::at_least::<T>(&b, &a)),
                Relation::Equal => wide::equal(&a, &b),
            };
        }
        compare_wires(self.wire(), relation, right.wire())
    }
    fn equal(self, right: Val<T>, equal: bool) -> Val<bool> {
        if let Some((a, b)) = wide_pair(self, right) {
            let bit = wide::equal(&a, &b);
            return if equal { bit } else { not_bit(bit) };
        }
        equality(self.wire(), right.wire(), equal)
    }
}

impl<T: Scalar> Cmp<T> for Val<T> {
    type Output = Val<bool>;
    fn compare(self, relation: Relation, right: T) -> Val<bool> {
        self.compare(relation, Val::constant(right))
    }
    fn equal(self, right: T, equal: bool) -> Val<bool> {
        Cmp::equal(self, Val::constant(right), equal)
    }
}

macro_rules! native_comparisons {
    ($($t:ident),*) => {$(
        impl Cmp<Val<$t>> for $t {
            type Output = Val<bool>;
            fn compare(self, relation: Relation, right: Val<$t>) -> Val<bool> {
                Val::constant(self).compare(relation, right)
            }
            fn equal(self, right: Val<$t>, equal: bool) -> Val<bool> {
                Cmp::equal(Val::constant(self), right, equal)
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

/// A comparison of two native scalars by their numeric values, e.g. an enum's
/// discriminants, which is how a derived `PartialOrd` orders variants.
pub fn compare_native<T: Scalar>(left: T, relation: Relation, right: T) -> bool {
    let (left, right) = (left.to_f64(), right.to_f64());
    match relation {
        Relation::Greater => left > right,
        Relation::AtLeast => left >= right,
        Relation::Less => left < right,
        Relation::AtMost => left <= right,
        Relation::Equal => left == right,
    }
}

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

pub(super) fn and_bits(left: Val<bool>, right: Val<bool>) -> Val<bool> {
    match (known_bit(left), known_bit(right)) {
        (Some(false), _) | (_, Some(false)) => return Val::constant(false),
        (Some(true), _) => return right,
        (_, Some(true)) => return left,
        _ => {}
    }
    with_circuit(|circuit| Val::from_wire(emit_and(circuit, &[left.wire(), right.wire()])))
}

fn or_bits(left: Val<bool>, right: Val<bool>) -> Val<bool> {
    match (known_bit(left), known_bit(right)) {
        (Some(true), _) | (_, Some(true)) => return Val::constant(true),
        (Some(false), _) => return right,
        (_, Some(false)) => return left,
        _ => {}
    }
    with_circuit(|circuit| {
        let wire = emit_or(circuit, &[left.wire(), right.wire()]);
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
        equality(self.wire(), right.wire(), false)
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

// ---- Running sums ---------------------------------------------------------------------------------------------------

/// A value [`running_sums`] can add: native integers in the original function, integer circuit values in its trace.
pub trait RunningSum: Copy {
    /// Whether the sums are built as a prefix network (circuit values) rather than a loop (native values).
    const NETWORK: bool;
    fn add_pair(self, later: Self) -> Self;
}

macro_rules! native_running_sum {
    ($($int:ty),*) => {$(
        impl RunningSum for $int {
            const NETWORK: bool = false;
            #[track_caller]
            fn add_pair(self, later: $int) -> $int {
                self + later
            }
        }
    )*};
}
native_running_sum!(u8, u16, u32, u64, usize, i8, i16, i32, i64);

impl<T: super::IntScalar> RunningSum for Val<T> {
    const NETWORK: bool = true;
    #[track_caller]
    fn add_pair(self, later: Val<T>) -> Val<T> {
        let at = Location::caller();
        // Its own operation, so the network keeps its shape instead of being copied into wider sums.
        with_circuit(|circuit| {
            let wire = circuit.round_affine(self.wire() + later.wire());
            number(circuit, wire, at)
        })
    }
}

/// Every running total of `items`: `[a, a + b, a + b + c, ...]`, for integers. The original function adds them in a
/// loop; the circuit uses a Brent-Kung prefix network, about 2n additions in about 2 log2(n) layers instead of n - 1
/// in n - 1. Integer sums within ±2^24 are exact in any grouping, so both agree; f32 sums would round differently,
/// so they aren't accepted.
#[track_caller]
pub fn running_sums<T: RunningSum, const N: usize>(items: [T; N]) -> [T; N] {
    if T::NETWORK {
        let sums = crate::prefix::brent_kung_prefix(&items, |&earlier, &later| earlier.add_pair(later));
        std::array::from_fn(|index| sums[index])
    } else {
        let mut sums = items;
        for index in 1..N {
            sums[index] = sums[index - 1].add_pair(items[index]);
        }
        sums
    }
}
