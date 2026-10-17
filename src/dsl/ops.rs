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
use crate::numeric::{Range, lowest_power_of_two};
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

/// Both sides as bytes for a bitwise operator: where either is held only as bytes, or either has bytes and the
/// other's come free (it has them too, is a constant, or is one byte), since splitting bytes into bits is cheaper
/// than splitting a wire of up to 24 bits.
pub(super) fn byte_pair<T: Scalar>(left: Val<T>, right: Val<T>) -> Option<(Limbs, Limbs)> {
    let free = |value: Val<T>| {
        value.has_bytes()
            || with_circuit(|circuit| {
                let range = circuit.range(value.wire());
                circuit.constant_value(value.wire()).is_some() || range.min >= 0.0 && range.max < 256.0
            })
    };
    let bytes =
        left.is_wide() || right.is_wide() || (left.has_bytes() || right.has_bytes()) && free(left) && free(right);
    bytes.then(|| (wide::limbs_as::<T, T>(left), wide::limbs_as::<T, T>(right)))
}

/// Both sides as bytes where one-wire arithmetic on a value that came from bytes may leave ±2^24, bounded by `low`
/// and `high` over the operands' ranges: the bytes wrap like they would have.
fn spilled<T: Scalar>(
    left: Val<T>,
    right: Val<T>,
    low: impl Fn(Range, Range) -> f64,
    high: impl Fn(Range, Range) -> f64,
) -> Option<(Limbs, Limbs)> {
    if !(left.spill || right.spill) || !is_wide::<T>() {
        return None;
    }
    let (a, b) = with_circuit(|circuit| (circuit.range(left.wire()), circuit.range(right.wire())));
    let fits = low(a, b) > -EXACT_LIMIT && high(a, b) < EXACT_LIMIT;
    (!fits).then(|| (wide::limbs_as::<T, T>(left), wide::limbs_as::<T, T>(right)))
}

fn product_bounds(a: Range, b: Range) -> [f64; 4] {
    [a.min * b.min, a.min * b.max, a.max * b.min, a.max * b.max]
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
        if let Some((a, b)) = spilled(self, right, |a, b| a.min + b.min, |a, b| a.max + b.max) {
            return wide::add(&a, &b);
        }
        sum::<T>(self.wire() + right.wire()).spilling(self.spill || right.spill)
    }
}
impl<T: Num> Add<T> for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn add(self, right: T) -> Val<T> {
        if self.is_wide() || self.spill || beyond(right) {
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
        if let Some((a, b)) = spilled(self, right, |a, b| a.min - b.max, |a, b| a.max - b.min) {
            return wide::sub(&a, &b);
        }
        sum::<T>(self.wire() - right.wire()).spilling(self.spill || right.spill)
    }
}
impl<T: Num> Sub<T> for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn sub(self, right: T) -> Val<T> {
        if self.is_wide() || self.spill || beyond(right) {
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
        let low = |a, b| product_bounds(a, b).into_iter().fold(f64::INFINITY, f64::min);
        let high = |a, b| product_bounds(a, b).into_iter().fold(f64::NEG_INFINITY, f64::max);
        if let Some((a, b)) = spilled(self, right, low, high) {
            return wide::mul(&a, &b);
        }
        product::<T>(self.wire(), right.wire()).spilling(self.spill || right.spill)
    }
}
impl<T: Num> Mul<T> for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn mul(self, right: T) -> Val<T> {
        if self.is_wide() || self.spill || beyond(right) {
            return self * Val::constant(right);
        }
        sum(self.wire() * right.to_f64())
    }
}
impl<T: Num> Neg for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn neg(self) -> Val<T> {
        if self.is_wide() || self.spill {
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
        if let Some((quotient, _)) = long_division(self, right.to_i128(), at) {
            return quotient;
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
        if let Some((_, remainder)) = long_division(self, right.to_i128(), at) {
            return remainder;
        }
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
    fn div(self, right: Val<T>) -> Val<T> {
        divide_by_value(self, right, Location::caller(), false)
    }
}
impl<T: Num> Rem for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn rem(self, right: Val<T>) -> Val<T> {
        divide_by_value(self, right, Location::caller(), true)
    }
}

/// `value / divisor`, or `value % divisor` when `remainder` is set, for a divisor that depends on the inputs. Small
/// ranges read a table of every pair's result; otherwise long division over the dividend's bits, one comparison step
/// per quotient bit, on magnitudes with the signs taken back after, like Rust's division toward zero. Division by
/// zero panics in Rust, so its result here is unspecified.
fn divide_by_value<T: Num>(value: Val<T>, divisor: Val<T>, at: &Location, remainder: bool) -> Val<T> {
    if T::KIND == Kind::Float {
        panic!("{at}: f32 division is only supported by a power of two, not by a value that depends on the inputs");
    }
    if divisor.is_wide() {
        panic!("{at}: this {} divisor may leave ±2^24; divisors up to 2^23 are supported", T::NAME);
    }
    let range = with_circuit(|circuit| circuit.range(divisor.wire()));
    if let Some(constant) = with_circuit(|circuit| circuit.constant_value(divisor.wire())) {
        let constant = T::from_f64(constant);
        return if remainder { value % constant } else { value / constant };
    }
    let (low, high) = (range.min.ceil(), range.max.floor());
    let largest = high.max(-low);
    if largest.is_nan() || largest > wide::MAX_DIVISOR as f64 {
        panic!("{at}: this {} divisor may reach {low}..{high}; divisors up to 2^23 are supported", T::NAME);
    }
    let zero = Val::constant(T::from_f64(0.0));
    if low == 0.0 && high == 0.0 {
        // Always a division by zero.
        return zero;
    }
    if !value.is_wide() {
        let dividend = with_circuit(|circuit| circuit.range(value.wire()));
        let (first, last) = (dividend.min.ceil(), dividend.max.floor());
        if dividend.is_finite() && (last - first + 1.0) * (high - low + 1.0) <= 65_536.0 {
            return pair_table(value, divisor, (first as i64, last as i64), (low as i64, high as i64), at, remainder);
        }
    }
    let signed = T::MIN < 0.0;
    let may_be_negative = value.has_bytes() || with_circuit(|circuit| circuit.range(value.wire()).min < 0.0);
    let value_negative = (signed && may_be_negative).then(|| lt(value, zero));
    let divisor_negative = (low < 0.0).then(|| lt(divisor, zero));
    let magnitude = match value_negative {
        Some(negative) => Val::select_by(negative, value, zero - value),
        None => value,
    };
    let divisor_magnitude = match divisor_negative {
        Some(negative) => Val::select_by(negative, divisor, zero - divisor),
        None => divisor,
    };
    let (d_min, d_max) = if low >= 0.0 {
        (low, high)
    } else if high <= 0.0 {
        (-high, -low)
    } else {
        (0.0, largest)
    };
    with_circuit(|circuit| settle::<T>(circuit, divisor_magnitude.wire(), d_min, d_max));
    let bits: Vec<Val<bool>> = if magnitude.is_wide() {
        wide::bits_of(&wide::limbs_as::<T, T>(magnitude))
    } else {
        if value_negative.is_some() {
            let range = with_circuit(|circuit| circuit.range(value.wire()));
            with_circuit(|circuit| settle::<T>(circuit, magnitude.wire(), 0.0, range.max.max(-range.min)));
        }
        let split = super::bits::split(magnitude);
        (0..split.len()).map(|k| split.bit(k)).collect()
    };
    let (quotient_bits, remainder_wire) = wide::divide_by_wire(&bits, divisor_magnitude.wire());
    if remainder {
        let result: Val<T> = Val::from_wire(remainder_wire);
        return match value_negative {
            Some(negative) => Val::select_by(negative, result, zero - result),
            None => result,
        };
    }
    let quotient: Val<T> = if magnitude.is_wide() {
        Val::from_limbs(wide::join_bits(&quotient_bits))
    } else {
        with_circuit(|circuit| {
            let mut form = Lin::new(vec![], 0.0);
            for (k, bit) in quotient_bits.iter().enumerate() {
                match circuit.constant_value(bit.wire()) {
                    Some(value) => form.bias += value * (1u64 << k) as f64,
                    None => form.terms.push((bit.wire(), (1u64 << k) as f64)),
                }
            }
            let wire = circuit.affine(form);
            number(circuit, wire, at)
        })
    };
    let flipped = match (value_negative, divisor_negative) {
        (Some(left), Some(right)) => Some(ne(left, right)),
        (Some(negative), None) | (None, Some(negative)) => Some(negative),
        (None, None) => None,
    };
    match flipped {
        Some(flipped) => Val::select_by(flipped, quotient, zero - quotient),
        None => quotient,
    }
}

/// `value / divisor` or `value % divisor` as one read of a table with every pair's result, for small ranges. Pairs
/// where Rust panics, dividing by zero or overflowing, read 0.
fn pair_table<T: Num>(
    value: Val<T>,
    divisor: Val<T>,
    (first, last): (i64, i64),
    (low, high): (i64, i64),
    at: &Location,
    remainder: bool,
) -> Val<T> {
    let mut results = Vec::new();
    for a in first..=last {
        for d in low..=high {
            let result = match (d, remainder) {
                (0, _) => 0,
                (_, false) => a / d,
                (_, true) => a % d,
            };
            results.push(if (T::MIN..=T::MAX).contains(&(result as f64)) { result } else { 0 });
        }
    }
    with_circuit(|circuit| {
        let table: Vec<Wire> = results.iter().map(|&result| circuit.constant(result as f64)).collect();
        let index = (value.wire() - first as f64) * (high - low + 1) as f64 + (divisor.wire() - low as f64);
        let read = circuit.index_read_affine(index, &table, OutOfRange::Clamp, OutOfRange::Clamp, None);
        circuit.assume_grid_allowing_readers(read, 1.0);
        number(circuit, read, at)
    })
}

/// Any integer function of one integer with few possible inputs, as one
/// table read: e.g. `x / 3` over x in 0..255 reads a table of 256 quotients.
/// `value / divisor` and `value % divisor` by long division over bytes, where a table over the value's range would
/// be too large: a wide value, or a one-wire value of a wide type with a range of 65,536 or more. Rust's division
/// truncates toward zero, so a signed value divides its magnitude and takes the signs back. Unsigned division by a
/// power of two is a shift and a mask.
fn long_division<T: Num>(value: Val<T>, divisor: i128, at: &Location) -> Option<(Val<T>, Val<T>)> {
    if T::KIND != Kind::Int || !is_wide::<T>() {
        return None;
    }
    if !value.has_bytes() {
        let range = with_circuit(|circuit| circuit.range(value.wire()));
        if range.is_finite() && range.max - range.min < 65_536.0 {
            return None;
        }
    }
    assert!(divisor != 0, "{at}: division by zero");
    let magnitude = divisor.unsigned_abs();
    if magnitude > wide::MAX_DIVISOR as u128 {
        panic!("{at}: dividing a value of a wide type by {divisor} isn't supported; divisors up to 2^23 are");
    }
    let zero = Val::constant(T::from_f64(0.0));
    if T::MIN >= 0.0 && magnitude.is_power_of_two() {
        let bytes = wide::limbs_as::<T, T>(value);
        let mask = wide::constant_limbs::<T>(magnitude as i128 - 1);
        let shift = magnitude.trailing_zeros() as usize;
        return Some((wide::shift(&bytes, shift, wide::Motion::Right), wide::bitwise(&bytes, &mask, 0.0)));
    }
    let negative = (T::MIN < 0.0).then(|| lt(value, zero));
    let unsigned = match negative {
        Some(negative) => Val::select_by(negative, value, zero - value),
        None => value,
    };
    let (quotient, remainder) = wide::divide_bytes(&wide::limbs_as::<T, T>(unsigned), magnitude as u64);
    let quotient: Val<T> = Val::from_limbs(quotient);
    let remainder: Val<T> = with_circuit(|circuit| settle(circuit, remainder, 0.0, (magnitude - 1) as f64));
    let Some(negative) = negative else { return Some((quotient, remainder)) };
    let flipped = if divisor < 0 { not_bit(negative) } else { negative };
    Some((Val::select_by(flipped, quotient, zero - quotient), Val::select_by(negative, remainder, zero - remainder)))
}

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
                Val::constant(self) / right
            }
        }
        impl Rem<Val<$t>> for $t {
            type Output = Val<$t>;
            #[track_caller]
            fn rem(self, right: Val<$t>) -> Val<$t> {
                Val::constant(self) % right
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
        if self.is_wide() {
            // Wrapping like the bytes do: i32::MIN stays i32::MIN, where Rust panics.
            let zero = Val::constant(T::from_f64(0.0));
            return Val::select_by(lt(self, zero), self, zero - self);
        }
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
