//! Integers beyond ±2^24, held as bytes.
//!
//! f32 holds every integer up to 2^24, so a `u32`, `i32`, `u64`, `i64` or
//! `usize` value stays one wire while its range fits there, as before. Where
//! a shift, a bitwise operator, a rotate, `wrapping_*`, a constant or an
//! operand that is already wide takes it beyond, it becomes its bytes: one
//! wire per 8 bits of the type's two's-complement pattern, each within
//! 0..=255. Plain `+`, `-` and `*` of one-wire values keep failing the build
//! where their range may leave ±2^24, so a loose `#[range]` isn't silently
//! turned into byte arithmetic.
//!
//! Bytes keep every step exact: a sum of two bytes and a carry is below 2^10,
//! a product of two bytes below 2^16, and a column of a product's bytes below
//! 2^19. Sums carry from byte to byte, one comparison step per carry bit;
//! products add their columns and carry the same way; comparisons are the
//! carry out of `a + !b + 1`; bitwise operators split bytes into bits; shifts
//! and rotates by multiples of 8 move whole bytes. Arithmetic wraps, like
//! `wrapping_*`, which is what Rust computes wherever it doesn't panic.
//! Parameters and results stay one cell within ±2^24: narrow a wide value,
//! e.g. with `as u8`, before returning it.

use std::panic::Location;

use super::bits::{integer_step, split};
use super::ops::equality;
use super::{Kind, Scalar, Val, known_bit, settle, width, with_circuit};
use crate::circuit::{Circuit, OutOfRange};
use crate::emit::emit_and;
use crate::wire::{Lin, Wire};

/// The most bytes a value has: 8, for 64-bit types.
const MAX_LIMBS: usize = 8;
const BASE: f64 = 256.0;

/// A wide value's bytes, least significant first.
#[derive(Copy, Clone)]
pub struct Limbs {
    wires: [Wire; MAX_LIMBS],
    count: usize,
}

impl Limbs {
    fn new(list: &[Wire]) -> Limbs {
        let mut wires = [list[0]; MAX_LIMBS];
        wires[..list.len()].copy_from_slice(list);
        Limbs { wires, count: list.len() }
    }

    pub fn wires(&self) -> &[Wire] {
        &self.wires[..self.count]
    }
}

/// The number of bytes of `T`, e.g. 4 for u32.
fn limb_count<T: Scalar>() -> usize {
    width::<T>() / 8
}

fn signed<T: Scalar>() -> bool {
    T::MIN < 0.0
}

fn constant(circuit: &mut Circuit, value: f64) -> Wire {
    let wire = circuit.constant(value);
    circuit.assume_grid_allowing_readers(wire, 1.0);
    wire
}

fn known(wire: Wire) -> Option<f64> {
    with_circuit(|circuit| circuit.constant_value(wire))
}

/// A byte from a sum known to lie in 0..=255: the wire itself for a lone
/// term, else its own sum, so readers read the byte instead of copying terms.
fn byte(circuit: &mut Circuit, form: Lin) -> Wire {
    if form.terms.is_empty() {
        return constant(circuit, form.bias);
    }
    if form.terms.len() == 1 && form.terms[0].1 == 1.0 && form.bias == 0.0 {
        return form.terms[0].0;
    }
    let wire = circuit.round_affine(form);
    circuit.assume_range_allowing_readers(wire, 0.0, 255.0);
    circuit.assume_grid_allowing_readers(wire, 1.0);
    wire
}

/// One wire for a value given as bytes, where its range fits within ±2^24:
/// the bytes' sum when the bytes above some position are known 0s (a
/// non-negative value), or, for signed types, known 255s (a negative value,
/// the sum minus 256^position).
pub(super) fn narrow_form<T: Scalar>(limbs: &Limbs) -> Option<Wire> {
    let wires = limbs.wires();
    let fill = |value: f64| wires.iter().rposition(|&wire| known(wire) != Some(value)).map_or(0, |last| last + 1);
    let (used, offset) = match fill(0.0) {
        used if used <= 3 => (used, 0.0),
        _ if signed::<T>() && fill(255.0) <= 3 => (fill(255.0), -BASE.powi(fill(255.0) as i32)),
        _ => return None,
    };
    with_circuit(|circuit| {
        let mut form = Lin::new(vec![], offset);
        let (mut low, mut high) = (offset, offset);
        for (index, &wire) in wires[..used].iter().enumerate() {
            let weight = BASE.powi(index as i32);
            let range = circuit.range(wire);
            low += range.min * weight;
            high += range.max * weight;
            match circuit.constant_value(wire) {
                Some(value) => form.bias += value * weight,
                None => form.terms.push((wire, weight)),
            }
        }
        if !(low > -16_777_216.0 && high < 16_777_216.0) {
            return None;
        }
        let wire = match form.terms.as_slice() {
            [] => constant(circuit, form.bias),
            [(wire, weight)] if *weight == 1.0 && form.bias == 0.0 => *wire,
            _ => circuit.affine(form),
        };
        Some(settle::<T>(circuit, wire, low, high).wire())
    })
}

/// The bytes of a constant's two's-complement pattern in `T`'s width.
pub(super) fn constant_limbs<T: Scalar>(value: i128) -> Limbs {
    let count = limb_count::<T>();
    let pattern = value.rem_euclid(1i128 << (8 * count));
    with_circuit(|circuit| {
        let wires: Vec<Wire> =
            (0..count).map(|index| constant(circuit, ((pattern >> (8 * index)) & 255) as f64)).collect();
        Limbs::new(&wires)
    })
}

/// The bytes of any value of `S` as a `T`: its own bytes resized, or a
/// one-wire value split into bits, sign-extended to `T`'s width, which is
/// also what `as` computes between integer types.
pub(super) fn limbs_as<S: Scalar, T: Scalar>(value: Val<S>) -> Limbs {
    let count = limb_count::<T>();
    if let Some(limbs) = value.wide {
        let wires = limbs.wires();
        if count <= wires.len() {
            return Limbs::new(&wires[..count]);
        }
        let fill = if signed::<S>() { sign_byte(wires[wires.len() - 1]) } else { constant_byte(0.0) };
        let mut extended = wires.to_vec();
        extended.resize(count, fill);
        return Limbs::new(&extended);
    }
    let wire = value.wire();
    if let Some(known) = known(wire) {
        return constant_limbs::<T>(known as i128);
    }
    let range = with_circuit(|circuit| circuit.range(wire));
    if range.min >= 0.0 && range.max < BASE {
        let zero = constant_byte(0.0);
        let mut wires = vec![zero; count];
        wires[0] = wire;
        return Limbs::new(&wires);
    }
    let bits = split(value);
    let wires: Vec<Wire> =
        (0..count).map(|index| join_byte(&(0..8).map(|k| bits.bit(8 * index + k)).collect::<Vec<_>>())).collect();
    Limbs::new(&wires)
}

fn constant_byte(value: f64) -> Wire {
    with_circuit(|circuit| constant(circuit, value))
}

/// 255 where a byte's top bit is set, else 0: the fill of a negative value.
fn sign_byte(top: Wire) -> Wire {
    with_circuit(|circuit| {
        let sign = integer_step(circuit, &Lin::new(vec![(top, 1.0)], 0.0), 128.0, true);
        byte(circuit, Lin::new(vec![(sign.wire(), 255.0)], 0.0))
    })
}

/// A byte from its 8 bits, least significant first.
fn join_byte(bits: &[Val<bool>]) -> Wire {
    with_circuit(|circuit| {
        let mut form = Lin::new(vec![], 0.0);
        for (k, bit) in bits.iter().enumerate() {
            match circuit.constant_value(bit.wire()) {
                Some(value) => form.bias += value * (1u32 << k) as f64,
                None => form.terms.push((bit.wire(), (1u32 << k) as f64)),
            }
        }
        byte(circuit, form)
    })
}

/// The 8 bits of each byte, least significant first.
pub(super) fn bits_of(limbs: &Limbs) -> Vec<Val<bool>> {
    let mut bits = Vec::with_capacity(8 * limbs.count);
    for &wire in limbs.wires() {
        let split = split(Val::<u8>::from_wire(wire));
        bits.extend((0..8).map(|k| split.bit(k)));
    }
    bits
}

pub(super) fn join_bits(bits: &[Val<bool>]) -> Limbs {
    let wires: Vec<Wire> = bits.chunks(8).map(join_byte).collect();
    Limbs::new(&wires)
}

// ---- Carries ----------------------------------------------------------------------------------------------------------

/// Split `form`, an integer within 0..=`largest`, into its low byte and the
/// carry, `form / 256` rounded down, as a sum of carry bits with its largest
/// value. `bytes` is false when only the carry is needed.
fn carry_out(circuit: &mut Circuit, form: Lin, largest: f64, bytes: bool) -> (Option<Wire>, Lin, f64) {
    let high = (largest / BASE).floor() as u64;
    let mut rest = form;
    let mut carry = Lin::new(vec![], 0.0);
    // Bits copied into readers stay exact while the ramps behind them stay small; see bits::COPIED_BITS.
    let copied = largest < 2048.0;
    let steps = 64 - high.leading_zeros();
    for step in (0..steps).rev() {
        let threshold = BASE * (1u64 << step) as f64;
        let bit = integer_step(circuit, &rest, threshold, copied);
        rest.terms.push((bit.wire(), -threshold));
        carry.terms.push((bit.wire(), (1u64 << step) as f64));
    }
    let low = bytes.then(|| byte(circuit, rest));
    (low, carry, high as f64)
}

/// Add per-byte sums with carries from byte to byte, each sum given with its
/// largest value; the last carry is dropped, so the result wraps. Returns the
/// bytes (when `bytes`) and the last carry.
fn carry_chain(forms: Vec<(Lin, f64)>, carry_in: f64, bytes: bool) -> (Vec<Wire>, Lin) {
    with_circuit(|circuit| {
        let mut carry = Lin::new(vec![], carry_in);
        let mut carry_largest = carry_in;
        let mut out = Vec::with_capacity(forms.len());
        for (mut form, largest) in forms {
            form.terms.extend(carry.terms.iter().copied());
            form.bias += carry.bias;
            let (low, next, next_largest) = carry_out(circuit, form, largest + carry_largest, bytes);
            out.extend(low);
            carry = next;
            carry_largest = next_largest;
        }
        (out, carry)
    })
}

fn sum_form(terms: &[(Wire, f64)], bias: f64) -> Lin {
    Lin::new(terms.to_vec(), bias)
}

// ---- Arithmetic -------------------------------------------------------------------------------------------------------

/// `a + b`, wrapping.
pub(super) fn add<T: Scalar>(a: &Limbs, b: &Limbs) -> Val<T> {
    let forms = (0..a.count).map(|i| (sum_form(&[(a.wires[i], 1.0), (b.wires[i], 1.0)], 0.0), 510.0)).collect();
    Val::from_limbs(Limbs::new(&carry_chain(forms, 0.0, true).0))
}

/// `a - b`, wrapping: `a + !b + 1`.
pub(super) fn sub<T: Scalar>(a: &Limbs, b: &Limbs) -> Val<T> {
    let forms = (0..a.count).map(|i| (sum_form(&[(a.wires[i], 1.0), (b.wires[i], -1.0)], 255.0), 510.0)).collect();
    Val::from_limbs(Limbs::new(&carry_chain(forms, 1.0, true).0))
}

/// `a * b`, wrapping: each column of byte products, then carries.
pub(super) fn mul<T: Scalar>(a: &Limbs, b: &Limbs) -> Val<T> {
    let count = a.count;
    let mut forms = Vec::with_capacity(count);
    for column in 0..count {
        let mut terms = Vec::new();
        let mut bias = 0.0;
        let mut largest = 0.0;
        for i in 0..=column {
            let (x, y) = (a.wires[i], b.wires[column - i]);
            match (known(x), known(y)) {
                (Some(p), Some(q)) => {
                    bias += p * q;
                    largest += p * q;
                }
                (Some(p), None) => {
                    terms.push((y, p));
                    largest += p * 255.0;
                }
                (None, Some(q)) => {
                    terms.push((x, q));
                    largest += q * 255.0;
                }
                (None, None) => {
                    let product = with_circuit(|circuit| circuit.multiply(x, y));
                    terms.push((product, 1.0));
                    largest += 65_025.0;
                }
            }
        }
        forms.push((sum_form(&terms, bias), largest));
    }
    Val::from_limbs(Limbs::new(&carry_chain(forms, 0.0, true).0))
}

/// `!a`: every bit flipped, 255 minus each byte.
pub(super) fn not<T: Scalar>(a: &Limbs) -> Val<T> {
    let wires: Vec<Wire> = with_circuit(|circuit| {
        a.wires().iter().map(|&wire| byte(circuit, Lin::new(vec![(wire, -1.0)], 255.0))).collect()
    });
    Val::from_limbs(Limbs::new(&wires))
}

// ---- Comparisons ------------------------------------------------------------------------------------------------------

/// `a >= b`: the carry out of `a + !b + 1`. Signed values compare with their
/// top bits flipped, which orders two's complement like unsigned.
pub(super) fn at_least<T: Scalar>(a: &Limbs, b: &Limbs) -> Val<bool> {
    let count = a.count;
    let top = |wire: Wire| -> Lin {
        if !signed::<T>() {
            return Lin::new(vec![(wire, 1.0)], 0.0);
        }
        // top ^ 128 = top + 128 - 256 × [top >= 128].
        with_circuit(|circuit| {
            let high = integer_step(circuit, &Lin::new(vec![(wire, 1.0)], 0.0), 128.0, true);
            Lin::new(vec![(wire, 1.0), (high.wire(), -BASE)], 128.0)
        })
    };
    let mut forms = Vec::with_capacity(count);
    for i in 0..count {
        let (x, y) = if i + 1 == count {
            (top(a.wires[i]), top(b.wires[i]))
        } else {
            (Lin::new(vec![(a.wires[i], 1.0)], 0.0), Lin::new(vec![(b.wires[i], 1.0)], 0.0))
        };
        let mut form = x;
        form.terms.extend(y.terms.iter().map(|&(wire, weight)| (wire, -weight)));
        form.bias += 255.0 - y.bias;
        forms.push((form, 510.0));
    }
    let (_, carry) = carry_chain(forms, 1.0, false);
    with_circuit(|circuit| {
        let wire = if carry.terms.is_empty() { constant(circuit, carry.bias) } else { circuit.affine(carry) };
        settle::<bool>(circuit, wire, 0.0, 1.0)
    })
}

/// `a == b`: every byte equal.
pub(super) fn equal(a: &Limbs, b: &Limbs) -> Val<bool> {
    let mut bits = Vec::with_capacity(a.count);
    for i in 0..a.count {
        let (x, y) = (a.wires[i], b.wires[i]);
        if x == y {
            continue;
        }
        let bit = equality(x, y, true);
        if known_bit(bit) == Some(false) {
            return Val::constant(false);
        }
        if known_bit(bit) != Some(true) {
            bits.push(bit.wire());
        }
    }
    match bits.len() {
        0 => Val::constant(true),
        1 => Val::from_wire(bits[0]),
        _ => with_circuit(|circuit| {
            let wire = emit_and(circuit, &bits);
            settle::<bool>(circuit, wire, 0.0, 1.0)
        }),
    }
}

// ---- Bitwise ----------------------------------------------------------------------------------------------------------

/// `a & b` at `weight` 0, `a | b` at 1, `a ^ b` at 2, byte by byte: `x + y - weight × (x & y)`. A constant byte of 0
/// or 255 needs no bits.
pub(super) fn bitwise<T: Scalar>(a: &Limbs, b: &Limbs, weight: f64) -> Val<T> {
    let mut wires = Vec::with_capacity(a.count);
    for i in 0..a.count {
        let (x, y) = (a.wires[i], b.wires[i]);
        let (p, q) = (known(x), known(y));
        let shortcut = |constant: f64, other: Wire| -> Option<Lin> {
            let other_form = Lin::new(vec![(other, 1.0)], 0.0);
            match (weight, constant) {
                (_, 0.0) if weight == 0.0 => Some(Lin::new(vec![], 0.0)),
                (_, 0.0) => Some(other_form),
                (_, 255.0) if weight == 0.0 => Some(other_form),
                (_, 255.0) if weight == 1.0 => Some(Lin::new(vec![], 255.0)),
                (_, 255.0) => Some(Lin::new(vec![(other, -1.0)], 255.0)),
                _ => None,
            }
        };
        let form = match (p, q) {
            (Some(p), Some(q)) => {
                let (p, q) = (p as u32, q as u32);
                let value = match weight as u32 {
                    0 => p & q,
                    1 => p | q,
                    _ => p ^ q,
                };
                Some(Lin::new(vec![], value as f64))
            }
            (Some(p), None) => shortcut(p, y),
            (None, Some(q)) => shortcut(q, x),
            (None, None) => None,
        };
        let wire = match form {
            Some(form) => with_circuit(|circuit| byte(circuit, form)),
            None => {
                let (left, right) = (split(Val::<u8>::from_wire(x)), split(Val::<u8>::from_wire(y)));
                let and: Vec<Val<bool>> = (0..8).map(|k| super::ops::and_bits(left.bit(k), right.bit(k))).collect();
                let and = join_byte(&and);
                if weight == 0.0 {
                    and
                } else {
                    with_circuit(|circuit| byte(circuit, Lin::new(vec![(x, 1.0), (y, 1.0), (and, -weight)], 0.0)))
                }
            }
        };
        wires.push(wire);
    }
    Val::from_limbs(Limbs::new(&wires))
}

/// The number of 1 bits.
pub(super) fn count_ones(a: &Limbs) -> Val<u32> {
    let bits = bits_of(a);
    let total = (8 * a.count) as f64;
    with_circuit(|circuit| {
        let mut form = Lin::new(vec![], 0.0);
        for bit in &bits {
            match circuit.constant_value(bit.wire()) {
                Some(value) => form.bias += value,
                None => form.terms.push((bit.wire(), 1.0)),
            }
        }
        let wire = if form.terms.is_empty() { constant(circuit, form.bias) } else { circuit.round_affine(form) };
        settle(circuit, wire, 0.0, total)
    })
}

/// How a shift or rotate moves bits.
#[derive(Copy, Clone, PartialEq)]
pub(super) enum Motion {
    Left,
    Right,
    RotateLeft,
    RotateRight,
}

/// Shift or rotate by a constant amount below the width. Right shifts of signed types fill with the sign.
pub(super) fn shift<T: Scalar>(a: &Limbs, amount: usize, motion: Motion) -> Val<T> {
    let width = 8 * a.count;
    let amount = amount % width;
    // Where bit k of the result comes from, if anywhere.
    let source = |k: usize| -> Option<usize> {
        match motion {
            Motion::Left => k.checked_sub(amount),
            Motion::Right => {
                let from = k + amount;
                if from < width {
                    Some(from)
                } else if signed::<T>() {
                    Some(width - 1)
                } else {
                    None
                }
            }
            Motion::RotateLeft => Some((k + width - amount) % width),
            Motion::RotateRight => Some((k + amount) % width),
        }
    };
    let filled_with_sign = motion == Motion::Right && signed::<T>() && amount > 0;
    if amount.is_multiple_of(8) && !filled_with_sign {
        // Whole bytes move.
        let zero = constant_byte(0.0);
        let wires: Vec<Wire> = (0..a.count).map(|i| source(8 * i).map_or(zero, |from| a.wires[from / 8])).collect();
        return Val::from_limbs(Limbs::new(&wires));
    }
    let bits = bits_of(a);
    let zero = Val::<bool>::constant(false);
    let moved: Vec<Val<bool>> = (0..width).map(|k| source(k).map_or(zero, |from| bits[from])).collect();
    let limbs = join_bits(&moved);
    let kept = width - amount;
    if filled_with_sign && kept <= 24 {
        // The bytes above `kept` bits repeat the sign, which no byte shows as a known constant: the one wire is the
        // kept bits, the top one counting -2^(kept - 1).
        let wire = with_circuit(|circuit| {
            let mut form = Lin::new(vec![], 0.0);
            for (k, bit) in moved[..kept].iter().enumerate() {
                let weight = if k + 1 == kept { -((1u64 << k) as f64) } else { (1u64 << k) as f64 };
                match circuit.constant_value(bit.wire()) {
                    Some(value) => form.bias += value * weight,
                    None => form.terms.push((bit.wire(), weight)),
                }
            }
            let half = (1u64 << (kept - 1)) as f64;
            let wire = if form.terms.is_empty() { constant(circuit, form.bias) } else { circuit.affine(form) };
            settle::<T>(circuit, wire, -half, half - 1.0).wire()
        });
        return Val::both(wire, limbs);
    }
    Val::from_limbs(limbs)
}

// ---- Division by a constant -------------------------------------------------------------------------------------

/// The largest divisor long division handles: each partial remainder, below the divisor, shifted by at least one
/// bit and plus that bit, stays within 2^24.
pub(super) const MAX_DIVISOR: u64 = 1 << 23;

/// `a / divisor` and `a % divisor` for `a`'s bytes read as unsigned, by long division: digits of up to 8 bits from
/// the top, each `remainder × 2^bits + digit` divided by comparison steps, one per quotient bit, like `carry_out`.
/// Returns the quotient's bytes and the remainder as one wire.
pub(super) fn divide_bytes(a: &Limbs, divisor: u64) -> (Limbs, Wire) {
    assert!((1..=MAX_DIVISOR).contains(&divisor));
    let d = divisor as f64;
    // Digits as wide as keep `remainder × 2^bits + digit` below 2^24.
    let digit_bits = (0..=8).rev().find(|&bits| (divisor - 1) * (1 << bits) + (1 << bits) - 1 < 1 << 24).unwrap();
    let width = 8 * a.count;
    let bits = if digit_bits == 8 { Vec::new() } else { bits_of(a) };
    let mut quotient = vec![Val::<bool>::constant(false); width];
    let mut remainder: Option<Wire> = None;
    // The largest the remainder can be so far: the value read so far, until that reaches the divisor.
    let mut remainder_max = 0.0;
    let mut top = width;
    while top > 0 {
        let low = top.saturating_sub(digit_bits);
        let size = top - low;
        let remainder_now = remainder;
        let wire = with_circuit(|circuit| {
            let mut form = Lin::new(vec![], 0.0);
            if let Some(remainder) = remainder_now {
                form.terms.push((remainder, (1u64 << size) as f64));
            }
            if digit_bits == 8 && low % 8 == 0 && size == 8 {
                form.terms.push((a.wires[low / 8], 1.0));
            } else {
                for (k, bit) in bits[low..top].iter().enumerate() {
                    match circuit.constant_value(bit.wire()) {
                        Some(value) => form.bias += value * (1u64 << k) as f64,
                        None => form.terms.push((bit.wire(), (1u64 << k) as f64)),
                    }
                }
            }
            let largest = remainder_max * (1u64 << size) as f64 + ((1u64 << size) - 1) as f64;
            let high = (largest / d).floor() as u64;
            let copied = largest < 2048.0;
            let mut rest = form;
            for step in (0..64 - high.leading_zeros()).rev() {
                let threshold = d * (1u64 << step) as f64;
                let bit = integer_step(circuit, &rest, threshold, copied);
                rest.terms.push((bit.wire(), -threshold));
                quotient[low + step as usize] = bit;
            }
            remainder_max = largest.min(d - 1.0);
            let wire = if rest.terms.is_empty() { constant(circuit, rest.bias) } else { circuit.round_affine(rest) };
            circuit.assume_range_allowing_readers(wire, 0.0, remainder_max);
            circuit.assume_grid_allowing_readers(wire, 1.0);
            wire
        });
        remainder = Some(wire);
        top = low;
    }
    (join_bits(&quotient), remainder.unwrap())
}

// ---- Division by a value ----------------------------------------------------------------------------------------

/// `a / divisor` and `a % divisor` for an unsigned dividend given by its bits, least significant first, and a divisor
/// wire within 0..=`MAX_DIVISOR`, by long division one bit at a time: each step doubles the remainder, adds the next
/// bit, and takes the divisor off where it fits. Steps whose remainder can't reach the divisor's smallest value are
/// known zeros and cost nothing. Returns the quotient's bits and the remainder.
pub(super) fn divide_by_wire(bits: &[Val<bool>], divisor: Wire) -> (Vec<Val<bool>>, Wire) {
    let range = with_circuit(|circuit| circuit.range(divisor));
    let (d_min, d_max) = (range.min.max(0.0), range.max);
    assert!(d_max <= MAX_DIVISOR as f64);
    let mut quotient = vec![Val::<bool>::constant(false); bits.len()];
    let mut rest = Lin::new(vec![], 0.0);
    // The largest the remainder can be so far: the value read so far, until that reaches the divisor.
    let mut largest = 0.0;
    for k in (0..bits.len()).rev() {
        let bit = bits[k];
        with_circuit(|circuit| {
            rest = rest.clone() * 2.0;
            match circuit.constant_value(bit.wire()) {
                Some(value) => rest.bias += value,
                None => rest.terms.push((bit.wire(), 1.0)),
            }
            let may_be_set = circuit.constant_value(bit.wire()).is_none_or(|value| value != 0.0);
            largest = 2.0 * largest + if may_be_set { 1.0 } else { 0.0 };
            if largest < d_min {
                return;
            }
            let fits = integer_step(circuit, &(rest.clone() - divisor), 0.0, false);
            let taken = circuit.multiply(fits.wire(), divisor);
            let remainder = circuit.round_affine(rest.clone() - taken);
            largest = largest.min((d_max - 1.0).max(0.0));
            circuit.assume_range_allowing_readers(remainder, 0.0, largest);
            circuit.assume_grid_allowing_readers(remainder, 1.0);
            rest = Lin::from(remainder);
            quotient[k] = fits;
        });
    }
    let remainder = with_circuit(|circuit| {
        if rest.terms.is_empty() {
            constant(circuit, rest.bias)
        } else if rest.terms.len() == 1 && rest.terms[0].1 == 1.0 && rest.bias == 0.0 {
            rest.terms[0].0
        } else {
            let wire = circuit.round_affine(rest);
            circuit.assume_range_allowing_readers(wire, 0.0, largest);
            circuit.assume_grid_allowing_readers(wire, 1.0);
            wire
        }
    });
    (quotient, remainder)
}

// ---- Casts, selects and reads ---------------------------------------------------------------------------------------

/// A wide value as a type of at most 16 bits: its low bytes, minus 2^width where a signed target's top bit is set.
#[track_caller]
pub(super) fn narrow<T: Scalar>(a: &Limbs) -> Val<T> {
    let at = Location::caller();
    if T::KIND == Kind::Float {
        panic!("{at}: casting an integer beyond ±2^24 to f32 rounds, which isn't supported yet");
    }
    let bytes = width::<T>() / 8;
    with_circuit(|circuit| {
        let mut form = Lin::new(vec![], 0.0);
        for (index, &wire) in a.wires()[..bytes].iter().enumerate() {
            form.terms.push((wire, BASE.powi(index as i32)));
        }
        if signed::<T>() {
            let top = a.wires[bytes - 1];
            let high = integer_step(circuit, &Lin::new(vec![(top, 1.0)], 0.0), 128.0, true);
            form.terms.push((high.wire(), -BASE.powi(bytes as i32)));
        }
        let wire = circuit.round_affine(form);
        circuit.assume_grid_allowing_readers(wire, 1.0);
        settle(circuit, wire, T::MIN, T::MAX)
    })
}

pub(super) fn select<T: Scalar>(bit: Val<bool>, off: &Limbs, on: &Limbs) -> Val<T> {
    let wires: Vec<Wire> = with_circuit(|circuit| {
        (0..off.count)
            .map(|i| {
                if off.wires[i] == on.wires[i] {
                    return off.wires[i];
                }
                let wire = circuit.select(bit.wire(), off.wires[i], on.wires[i]);
                circuit.assume_range_allowing_readers(wire, 0.0, 255.0);
                circuit.assume_grid_allowing_readers(wire, 1.0);
                wire
            })
            .collect()
    });
    Val::from_limbs(Limbs::new(&wires))
}

pub(super) fn read<T: Scalar>(items: &[Limbs], index: Wire) -> Val<T> {
    let count = items[0].count;
    let wires: Vec<Wire> = with_circuit(|circuit| {
        (0..count)
            .map(|i| {
                let candidates: Vec<Wire> = items.iter().map(|item| item.wires[i]).collect();
                let wire = circuit.index_read(index, &candidates, OutOfRange::Clamp);
                circuit.assume_range_allowing_readers(wire, 0.0, 255.0);
                circuit.assume_grid_allowing_readers(wire, 1.0);
                wire
            })
            .collect()
    });
    Val::from_limbs(Limbs::new(&wires))
}
