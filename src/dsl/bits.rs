//! Integer bitwise operators and shifts on circuit values.
//!
//! An integer is split into bits once: in two's complement, `low` bits plus a
//! sign bit that repeats above them, so `value = Σ low[k] 2^k - sign 2^len`.
//! A value that can't be negative has a constant sign of 0. The split takes
//! one comparison per bit, highest first, so it is `len` layers deep.
//!
//! Every bitwise operator reduces to AND, one ReLU per bit position:
//! `a ^ b = a + b - 2 (a & b)` and `a | b = a + b - (a & b)` hold for all
//! integers, so they are sums over the AND's result. A shift by a constant
//! moves bits: `>>` keeps the high ones, which is a floor division in Rust's
//! arithmetic shift, and `<<` wraps into the type's width only when the
//! shifted value may leave the type. A shift by a circuit value reads the
//! result for every possible amount at that amount.

use std::ops::{BitAnd, BitOr, BitXor, Not};
use std::panic::Location;

use super::ops::{and_bits, number};
use super::{IntScalar, Mux, Scalar, Val, known_bit, settle, with_circuit};
use crate::circuit::Circuit;
use crate::wire::{Lin, Wire};

/// The bits of an integer in two's complement.
struct Bits {
    low: Vec<Val<bool>>,
    sign: Val<bool>,
}

impl Bits {
    /// Bit `k`, the sign above the low bits.
    fn bit(&self, k: usize) -> Val<bool> {
        self.low.get(k).copied().unwrap_or(self.sign)
    }
}

/// The type's width in bits, e.g. 8 for u8 and i8.
fn width<T: Scalar>() -> usize {
    let magnitude = (T::MAX + 1.0).log2() as usize;
    if T::MIN < 0.0 { magnitude + 1 } else { magnitude }
}

fn constant_bit(value: bool) -> Val<bool> {
    Val::constant(value)
}

fn integer_constant(circuit: &mut Circuit, value: f64) -> Wire {
    let wire = circuit.constant(value);
    circuit.assume_grid_allowing_readers(wire, 1.0);
    wire
}

/// The bits of a native integer.
fn native_bits(value: i64) -> Bits {
    let mut len = 0;
    while !(-(1i64 << len) <= value && value < (1i64 << len)) {
        len += 1;
    }
    let low = (0..len).map(|k| constant_bit((value >> k) & 1 == 1)).collect();
    Bits { low, sign: constant_bit(value < 0) }
}

/// Values of up to this many bits are split with bits that readers copy into
/// their own sums: the ramps behind them then reach 2^(2 len + 1) at most,
/// within 2^23, so f32 adds them exactly in any order.
const COPIED_BITS: usize = 11;

/// `[form >= threshold]` for an integer form: `relu(form - threshold + 1) -
/// relu(form - threshold)`, two ReLUs side by side. A `copied` bit is an
/// ordinary sum that readers fold into theirs, so each bit is one layer;
/// otherwise it is kept as its own sum, which readers read as a 0 or 1, since
/// copying ramps of up to 2^24 times 2^k would round.
fn integer_step(circuit: &mut Circuit, form: &Lin, threshold: f64, copied: bool) -> Val<bool> {
    let shifted = |offset: f64| Lin::new(form.terms.clone(), form.bias + offset);
    let upper = circuit.relu(shifted(1.0 - threshold));
    let lower = circuit.relu(shifted(-threshold));
    let difference = Lin::new(vec![(upper, 1.0), (lower, -1.0)], 0.0);
    let bit = if copied { circuit.affine(difference) } else { circuit.round_affine(difference) };
    settle::<bool>(circuit, bit, 0.0, 1.0)
}

/// The bits of a circuit integer, from its range. The rest left to split
/// stays a sum, so each bit's step reads it directly: one layer per bit.
fn split<T: Scalar>(value: Val<T>) -> Bits {
    if let Some(known) = with_circuit(|circuit| circuit.constant_value(value.wire)) {
        return native_bits(known as i64);
    }
    with_circuit(|circuit| {
        let range = circuit.range(value.wire);
        let (low, high) = (range.min.ceil(), range.max.floor());
        assert!(
            low > -16_777_216.0 && high < 16_777_216.0,
            "bitwise operators need integers within ±2^24, and this {} may reach {low}..{high}",
            T::NAME
        );
        let mut len = 0;
        while !(-((1u64 << len) as f64) <= low && high < (1u64 << len) as f64) {
            len += 1;
        }
        let copied = len <= COPIED_BITS;
        let mut rest = Lin::new(vec![(value.wire, 1.0)], 0.0);
        // A negative value is brought into 0..2^len: rest = value + 2^len where value < 0.
        let sign = if low < 0.0 {
            let negated = Lin::new(vec![(value.wire, -1.0)], 0.0);
            let sign = integer_step(circuit, &negated, 1.0, copied);
            rest.terms.push((sign.wire, (1u64 << len) as f64));
            sign
        } else {
            Val::<bool>::from_wire(integer_constant(circuit, 0.0))
        };
        let mut low_bits = vec![sign; len];
        for k in (0..len).rev() {
            let bit = if k == 0 {
                // What's left is 0 or 1: the last bit itself.
                let wire = if copied { circuit.affine(rest.clone()) } else { circuit.round_affine(rest.clone()) };
                circuit.assume_range_allowing_readers(wire, 0.0, 1.0);
                settle::<bool>(circuit, wire, 0.0, 1.0)
            } else {
                integer_step(circuit, &rest, (1u64 << k) as f64, copied)
            };
            low_bits[k] = bit;
            rest.terms.push((bit.wire, -((1u64 << k) as f64)));
        }
        Bits { low: low_bits, sign }
    })
}

/// `extra + scale × (Σ low[k] 2^k - sign 2^len)` as one sum, whose value has
/// `low.len()` bits and, where `negative`, a sign: e.g. `left + right - 2
/// (left & right)` for XOR.
fn join<T: Scalar>(
    low: &[Val<bool>],
    sign: Val<bool>,
    scale: f64,
    extra: Lin,
    negative: bool,
    at: &Location,
) -> Val<T> {
    let mut form = extra;
    for (k, bit) in low.iter().enumerate() {
        form.terms.push((bit.wire, scale * (1u64 << k) as f64));
    }
    let top = (1u64 << low.len()) as f64;
    form.terms.push((sign.wire, -scale * top));
    with_circuit(|circuit| {
        // Kept as its own sum, so that readers, e.g. `(a ^ b) * 4096`, read the result instead of copying the bits'
        // ramps into sums where f32 would round them.
        let wire = circuit.round_affine(form);
        // The sum's own range adds its terms' ranges; the bits bound it tighter.
        circuit.assume_range_allowing_readers(wire, if negative { -top } else { 0.0 }, top - 1.0);
        circuit.assume_grid_allowing_readers(wire, 1.0);
        let joined: Val<T> = number(circuit, wire, at);
        settle(circuit, joined.wire, T::MIN, T::MAX)
    })
}

fn may_be_set(bit: Val<bool>) -> bool {
    known_bit(bit) != Some(false)
}

/// `left & right` when `weight` is 0, `left | right` at 1 and `left ^ right` at 2: `left + right - weight × (left &
/// right)`, which holds for all integers in two's complement.
#[track_caller]
fn bitwise<T: Scalar>(left: Val<T>, right: Val<T>, weight: f64) -> Val<T> {
    let at = Location::caller();
    let (left_bits, right_bits) = (split(left), split(right));
    let len = left_bits.low.len().max(right_bits.low.len());
    let low: Vec<Val<bool>> = (0..len).map(|k| and_bits(left_bits.bit(k), right_bits.bit(k))).collect();
    let sign = and_bits(left_bits.sign, right_bits.sign);
    if weight == 0.0 {
        return join(&low, sign, 1.0, Lin::new(vec![], 0.0), may_be_set(sign), at);
    }
    // OR and XOR are negative where either side is.
    let negative = may_be_set(left_bits.sign) || may_be_set(right_bits.sign);
    join(&low, sign, -weight, Lin::new(vec![(left.wire, 1.0), (right.wire, 1.0)], 0.0), negative, at)
}

/// `value << amount` for a native amount, wrapped into the type's width like Rust.
fn shift_left<T: Scalar>(value: Val<T>, amount: u32, at: &Location) -> Val<T> {
    let factor = 2f64.powi(amount as i32);
    let range = with_circuit(|circuit| circuit.range(value.wire));
    if range.min * factor >= T::MIN && range.max * factor <= T::MAX {
        return with_circuit(|circuit| {
            let wire = circuit.affine(value.wire * factor);
            number(circuit, wire, at)
        });
    }
    let bits = split(value);
    let width = width::<T>();
    let shifted = |position: usize| {
        if position < amount as usize { constant_bit(false) } else { bits.bit(position - amount as usize) }
    };
    if T::MIN < 0.0 {
        let low: Vec<Val<bool>> = (0..width - 1).map(shifted).collect();
        join(&low, shifted(width - 1), 1.0, Lin::new(vec![], 0.0), may_be_set(shifted(width - 1)), at)
    } else {
        let low: Vec<Val<bool>> = (0..width).map(shifted).collect();
        join(&low, constant_bit(false), 1.0, Lin::new(vec![], 0.0), false, at)
    }
}

/// `value >> amount` for a native amount: the high bits, a floor division by 2^amount.
fn shift_right<T: Scalar>(value: Val<T>, amount: u32, at: &Location) -> Val<T> {
    let bits = split(value);
    let low: Vec<Val<bool>> = bits.low.iter().skip(amount as usize).copied().collect();
    join(&low, bits.sign, 1.0, Lin::new(vec![], 0.0), may_be_set(bits.sign), at)
}

#[track_caller]
fn shift_native<T: Scalar>(value: Val<T>, amount: i64, left: bool) -> Val<T> {
    let at = Location::caller();
    let width = width::<T>() as i64;
    assert!((0..width).contains(&amount), "{at}: shifting a {} by {amount} overflows, which panics in Rust", T::NAME);
    if left { shift_left(value, amount as u32, at) } else { shift_right(value, amount as u32, at) }
}

/// A shift by a circuit amount: the result for each amount the range allows, read at the amount.
#[track_caller]
fn shift_by<T: Scalar, I: IntScalar>(value: Val<T>, amount: Val<I>, left: bool) -> Val<T> {
    let at = Location::caller();
    let range = with_circuit(|circuit| circuit.range(amount.wire));
    // Amounts outside 0..width panic in Rust, so only those within it need results.
    let last = (range.max.floor() as i64).min(width::<T>() as i64 - 1).max(0);
    let results: Vec<Val<T>> = (0..=last)
        .map(|shift| if left { shift_left(value, shift as u32, at) } else { shift_right(value, shift as u32, at) })
        .collect();
    Mux::read_dynamic(&results, amount)
}

// ---- Operators --------------------------------------------------------------------------------------------------------

impl<T: IntScalar> BitAnd for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn bitand(self, right: Val<T>) -> Val<T> {
        bitwise(self, right, 0.0)
    }
}
impl<T: IntScalar> BitOr for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn bitor(self, right: Val<T>) -> Val<T> {
        bitwise(self, right, 1.0)
    }
}
impl<T: IntScalar> BitXor for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn bitxor(self, right: Val<T>) -> Val<T> {
        bitwise(self, right, 2.0)
    }
}
impl<T: IntScalar> BitAnd<T> for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn bitand(self, right: T) -> Val<T> {
        bitwise(self, Val::constant(right), 0.0)
    }
}
impl<T: IntScalar> BitOr<T> for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn bitor(self, right: T) -> Val<T> {
        bitwise(self, Val::constant(right), 1.0)
    }
}
impl<T: IntScalar> BitXor<T> for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn bitxor(self, right: T) -> Val<T> {
        bitwise(self, Val::constant(right), 2.0)
    }
}

/// `!x`: `-1 - x` in two's complement, `MAX - x` for unsigned types.
impl<T: IntScalar> Not for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn not(self) -> Val<T> {
        let at = Location::caller();
        let flip = if T::MIN < 0.0 { -1.0 } else { T::MAX };
        with_circuit(|circuit| {
            let wire = circuit.affine(flip - self.wire);
            number(circuit, wire, at)
        })
    }
}

macro_rules! native_left_bitwise {
    ($($t:ident),*) => {$(
        impl BitAnd<Val<$t>> for $t {
            type Output = Val<$t>;
            #[track_caller]
            fn bitand(self, right: Val<$t>) -> Val<$t> {
                bitwise(Val::constant(self), right, 0.0)
            }
        }
        impl BitOr<Val<$t>> for $t {
            type Output = Val<$t>;
            #[track_caller]
            fn bitor(self, right: Val<$t>) -> Val<$t> {
                bitwise(Val::constant(self), right, 1.0)
            }
        }
        impl BitXor<Val<$t>> for $t {
            type Output = Val<$t>;
            #[track_caller]
            fn bitxor(self, right: Val<$t>) -> Val<$t> {
                bitwise(Val::constant(self), right, 2.0)
            }
        }
    )*};
}
native_left_bitwise!(u8, u16, u32, u64, usize, i8, i16, i32, i64);

/// `<<` and `>>`: native on two natives, else on the circuit; the amount may be any integer type.
pub trait Shift<Amount> {
    type Output;
    fn shift(self, amount: Amount, left: bool) -> Self::Output;
}

impl<T: IntScalar, A: IntScalar> Shift<A> for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn shift(self, amount: A, left: bool) -> Val<T> {
        shift_native(self, amount.to_f64() as i64, left)
    }
}
impl<T: IntScalar, A: IntScalar> Shift<Val<A>> for Val<T> {
    type Output = Val<T>;
    #[track_caller]
    fn shift(self, amount: Val<A>, left: bool) -> Val<T> {
        shift_by(self, amount, left)
    }
}

macro_rules! native_shifts {
    ($($t:ident),*) => {$(
        impl<A: IntScalar> Shift<Val<A>> for $t {
            type Output = Val<$t>;
            #[track_caller]
            fn shift(self, amount: Val<A>, left: bool) -> Val<$t> {
                shift_by(Val::constant(self), amount, left)
            }
        }
        native_shifts!(@by $t: u8, u16, u32, u64, usize, i8, i16, i32, i64);
    )*};
    (@by $t:ident: $($a:ident),*) => {$(
        impl Shift<$a> for $t {
            type Output = $t;
            fn shift(self, amount: $a, left: bool) -> $t {
                if left { self << amount } else { self >> amount }
            }
        }
    )*};
}
native_shifts!(u8, u16, u32, u64, usize, i8, i16, i32, i64);

/// `value << amount`.
#[track_caller]
pub fn shl<V: Shift<A>, A>(value: V, amount: A) -> V::Output {
    value.shift(amount, true)
}

/// `value >> amount`.
#[track_caller]
pub fn shr<V: Shift<A>, A>(value: V, amount: A) -> V::Output {
    value.shift(amount, false)
}

impl<T: IntScalar> Val<T> {
    /// The number of 1 bits in the type's width, like `u8::count_ones`.
    #[track_caller]
    pub fn count_ones(self) -> Val<u32> {
        let bits = split(self);
        let width = width::<T>();
        let mut form = Lin::new(bits.low.iter().map(|bit| (bit.wire, 1.0)).collect(), 0.0);
        // The sign repeats in every bit above the low ones.
        form.terms.push((bits.sign.wire, (width - bits.low.len()) as f64));
        with_circuit(|circuit| {
            let wire = circuit.round_affine(form);
            settle(circuit, wire, 0.0, width as f64)
        })
    }
}
