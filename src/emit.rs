//! Builder functions ("macros") that expand familiar expressions into operations.
//!
//! Ported from handwired's `numeric-ui-core.ts`. Each function documents when
//! its result is exact; see handwired's docs/expressions.md for the cost tables.

use crate::circuit::{Circuit, OpKind, RoundingPointKind};
use crate::numeric::{fround, largest_grid, lowest_power_of_two, pow2, power_of_two_exponent_at_least, smallest_grid, Range, EXACT_GRID_STEPS};
use crate::wire::{Lin, Term, Wire};

/// `max(0, value)`.
pub fn emit_relu(circuit: &mut Circuit, value: Wire) -> Wire {
    circuit.reglu(1.0, value)
}

/// `max(a, b)` rebuilt through a rounded difference: `b + relu(a - b)`.
/// Not always exactly one of the operands; see [`emit_maximum_exact`].
pub fn emit_maximum(circuit: &mut Circuit, a: Wire, b: Wire) -> Wire {
    let difference = circuit.round_affine(a - b);
    let positive = emit_relu(circuit, difference);
    let result = circuit.round_affine(b + positive);
    let (a_range, b_range) = (circuit.range(a), circuit.range(b));
    if a_range.min >= 0.0 && a_range.max <= 1.0 && b_range.min >= 0.0 && b_range.max <= 1.0 {
        circuit.assume_range(result, 0.0, 1.0);
    }
    result
}

/// `min(a, b)` rebuilt through a rounded difference: `a - relu(a - b)`, e.g.
/// 0.29998779 for min(f(1000.37), f(0.3)). See [`emit_minimum_exact`].
pub fn emit_minimum(circuit: &mut Circuit, a: Wire, b: Wire) -> Wire {
    let difference = circuit.round_affine(a - b);
    let positive = emit_relu(circuit, difference);
    let result = circuit.round_affine(a - positive);
    if circuit.is_non_negative(a) && circuit.is_non_negative(b) && circuit.range(a).is_finite() && circuit.range(b).is_finite() {
        let maximum = fround(circuit.range(a).max);
        circuit.assume_range(result, 0.0, maximum);
    }
    result
}

/// `gate ? on : off` for a gate of exactly 0 or 1. Both sides always compute.
pub fn emit_select(circuit: &mut Circuit, gate: Wire, off: Wire, on: Wire) -> Wire {
    circuit.select(gate, off, on)
}

/// `!bit` for an exact bit.
pub fn emit_not(circuit: &mut Circuit, bit: Wire) -> Wire {
    circuit.affine(1.0 - bit)
}

/// `bits.all()`: `relu(sum - (n - 1))`, exact for exact bits.
pub fn emit_and(circuit: &mut Circuit, bits: &[Wire]) -> Wire {
    let sum = Lin::new(bits.iter().map(|&bit| (bit, 1.0)).collect(), 1.0 - bits.len() as f64);
    circuit.reglu(1.0, sum)
}

/// `bits.any()`: `1 - relu(1 - sum)`, exact for exact bits.
pub fn emit_or(circuit: &mut Circuit, bits: &[Wire]) -> Wire {
    let missing = Lin::new(bits.iter().map(|&bit| (bit, -1.0)).collect(), 1.0);
    let none = circuit.relu(missing);
    circuit.affine(1.0 - none)
}

/// `Math.round(x)` for |x| below 2^22: FP32 values within 2^23..2^24 are whole
/// numbers, so adding and then subtracting 2^23 + 2^22 rounds (ties to even).
/// Both steps are marked rounding points.
pub fn emit_round(circuit: &mut Circuit, value: Wire) -> Wire {
    const SHIFT: f64 = 12_582_912.0; // 2^23 + 2^22
    let shifted = circuit.round_affine(value + SHIFT);
    let rounded = circuit.round_affine(shifted - SHIFT);
    circuit.mark_rounding_point(shifted);
    circuit.mark_rounding_point(rounded);
    rounded
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Relation {
    Greater,
    AtLeast,
    Less,
    AtMost,
    Equal,
}

impl Relation {
    /// The relation as written in source, e.g. `">="`.
    pub fn symbol(self) -> &'static str {
        match self {
            Relation::Greater => ">",
            Relation::AtLeast => ">=",
            Relation::Less => "<",
            Relation::AtMost => "<=",
            Relation::Equal => "==",
        }
    }

    /// The comparison on native f32 values, the oracle every bit must match.
    pub fn holds(self, a: f32, b: f32) -> bool {
        match self {
            Relation::Greater => a > b,
            Relation::AtLeast => a >= b,
            Relation::Less => a < b,
            Relation::AtMost => a <= b,
            Relation::Equal => a == b,
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct CompareOptions {
    /// End with a boolean atom instead of a ReLU stage: fewer layers, but the
    /// executor must keep FP32 rounding at the atom, and backprop rejects it
    /// unless it is told to pass gradients straight through.
    pub end_with_boolean: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum SignTest {
    Greater,
    AtLeast,
    Equal,
}

#[derive(Copy, Clone, Debug)]
struct CompareOperand {
    wire: Option<Wire>,
    value: f64,
    range: Range,
    grid: Option<f64>,
}

fn compare_operand(circuit: &Circuit, operand: Wire) -> CompareOperand {
    circuit.assert_wire(operand);
    if let Some(constant) = circuit.constant_value(operand) {
        let value = fround(constant);
        assert!(value.is_finite(), "a constant comparison operand must be finite");
        // -0 and +0 compare equal, so a zero constant becomes +0.
        let value = if value == 0.0 { 0.0 } else { value };
        return CompareOperand {
            wire: None,
            value,
            range: Range::new(value, value),
            grid: Some(if value == 0.0 { f64::INFINITY } else { lowest_power_of_two(value) }),
        };
    }
    CompareOperand { wire: Some(operand), value: 0.0, range: circuit.range(operand), grid: circuit.grid(operand) }
}

/// `[a relation b]` as an exact bit, equal to the native f32 comparison for
/// every pair of values within the declared ranges. Both operands need finite
/// declared ranges. When both lie on grids, the bit is built from the integer
/// `D = (a - b) / grid` in 2 or 3 operations; otherwise a sign cascade lifts
/// every nonzero `f(a - b)`, down to 2^-149, to at least 1.
pub fn emit_compare(circuit: &mut Circuit, a: Wire, relation: Relation, b: Wire) -> Wire {
    emit_compare_with(circuit, a, relation, b, CompareOptions::default())
}

pub fn emit_compare_with(circuit: &mut Circuit, a: Wire, relation: Relation, b: Wire, options: CompareOptions) -> Wire {
    let mut left = compare_operand(circuit, a);
    let mut right = compare_operand(circuit, b);
    if matches!(relation, Relation::Less | Relation::AtMost) {
        std::mem::swap(&mut left, &mut right);
    }
    let test = match relation {
        Relation::Greater | Relation::Less => SignTest::Greater,
        Relation::AtLeast | Relation::AtMost => SignTest::AtLeast,
        Relation::Equal => SignTest::Equal,
    };
    if left.wire.is_none() && right.wire.is_none() {
        let holds = match test {
            SignTest::Greater => left.value > right.value,
            SignTest::AtLeast => left.value >= right.value,
            SignTest::Equal => left.value == right.value,
        };
        return circuit.constant(if holds { 1.0 } else { 0.0 });
    }
    for operand in [left, right] {
        assert!(
            operand.range.is_finite(),
            "emit_compare needs finite declared ranges on both operands, got {}..{}",
            operand.range.min,
            operand.range.max
        );
    }
    // A wire equals itself for every value its finite range allows.
    if left.wire == right.wire {
        return circuit.constant(if test == SignTest::Greater { 0.0 } else { 1.0 });
    }
    let difference_bound = (left.range.max - right.range.min).abs().max((left.range.min - right.range.max).abs());
    assert!(fround(difference_bound).is_finite(), "the difference of the declared operand ranges can overflow FP32");
    let mut terms: Vec<Term> = Vec::new();
    if let Some(wire) = left.wire {
        terms.push((wire, 1.0));
    }
    if let Some(wire) = right.wire {
        terms.push((wire, -1.0));
    }
    // At most one side is a constant, so the bias is exactly that constant or its negation.
    let bias = left.value - right.value;
    let bit = match emit_grid_compare(circuit, &left, &right, &terms, bias, test, options.end_with_boolean) {
        Some(bit) => bit,
        None => emit_cascade_compare(circuit, &terms, bias, fround(difference_bound), test, options.end_with_boolean),
    };
    circuit.assume_range(bit, 0.0, 1.0);
    circuit.assume_grid(bit, 1.0);
    if bit.index() >= circuit.source_size() {
        let index = bit.index() - circuit.source_size();
        circuit.operations[index].comparison_bit = true;
    }
    bit
}

/// The largest magnitude of any product or partial sum of bias + Σ coefficient
/// × operand, added in stored order, from the declared operand ranges.
fn largest_partial_sum(terms: &[Term], bias: f64, ranges: &[Range]) -> f64 {
    let (mut minimum, mut maximum, mut largest) = (bias, bias, bias.abs());
    for &(wire, coefficient) in terms {
        let range = ranges[wire.index()];
        let (low, high) =
            if coefficient >= 0.0 { (coefficient * range.min, coefficient * range.max) } else { (coefficient * range.max, coefficient * range.min) };
        minimum += low;
        maximum += high;
        largest = largest.max(low.abs()).max(high.abs()).max(minimum.abs()).max(maximum.abs());
    }
    largest
}

fn scaled_terms(terms: &[Term], factor: f64) -> Vec<Term> {
    terms.iter().map(|&(wire, coefficient)| (wire, coefficient * factor)).collect()
}

/// Integer bits for operands on grids. Returns `None` when an operand has no
/// grid or the ranges are too wide, and the caller builds a sign cascade.
fn emit_grid_compare(
    circuit: &mut Circuit,
    left: &CompareOperand,
    right: &CompareOperand,
    terms: &[Term],
    bias: f64,
    test: SignTest,
    end_with_boolean: bool,
) -> Option<Wire> {
    let grid = left.grid?.min(right.grid?);
    let largest_steps = |circuit: &Circuit, sign: f64, offset: f64| {
        largest_partial_sum(&scaled_terms(terms, sign / grid), sign * bias / grid + offset, &circuit.ranges)
    };
    let mut emitted = Vec::new();
    let bit;
    if end_with_boolean {
        // The score 0.5 + (D - offset) × 2^-24 is a multiple of 2^-24 within
        // 0.25..0.75 while every partial sum stays within 2^22 grid steps.
        let score_scale = pow2(-24) / grid;
        let fits = match test {
            SignTest::Equal => largest_steps(circuit, 1.0, 0.0) <= pow2(22) && largest_steps(circuit, -1.0, 0.0) <= pow2(22),
            _ => largest_steps(circuit, 1.0, if test == SignTest::Greater { -1.0 } else { 0.0 }) <= pow2(22),
        };
        if !(fits && score_scale >= smallest_grid() && score_scale <= largest_grid()) {
            return None;
        }
        let mut emit_atom = |circuit: &mut Circuit, sign: f64, offset: f64| {
            let score = Lin::new(scaled_terms(terms, sign * score_scale), 0.5 + (sign * bias / grid - offset) * pow2(-24));
            let atom = circuit.append(OpKind::Boolean, score, Lin::default(), true, None);
            emitted.push(atom);
            atom
        };
        if test == SignTest::Equal {
            let at_least = emit_atom(circuit, 1.0, 0.0);
            let at_most = emit_atom(circuit, -1.0, 0.0);
            bit = circuit.reglu(1.0, at_least + at_most - 1.0);
            emitted.push(bit);
        } else {
            bit = emit_atom(circuit, 1.0, if test == SignTest::Greater { 1.0 } else { 0.0 });
        }
    } else {
        let scale = 1.0 / grid;
        // The bit's own sum, 1 - ramp, adds one more step.
        let fits = match test {
            SignTest::Equal => {
                largest_steps(circuit, 1.0, 0.0) < EXACT_GRID_STEPS && largest_steps(circuit, -1.0, 0.0) < EXACT_GRID_STEPS
            }
            _ => largest_steps(circuit, -1.0, if test == SignTest::Greater { 1.0 } else { 0.0 }) < EXACT_GRID_STEPS,
        };
        if !(fits && scale <= largest_grid()) {
            return None;
        }
        bit = emit_integer_bit(circuit, test, &mut emitted, |circuit, sign, offset| {
            let ramp = Lin::new(scaled_terms(terms, sign * scale), sign * bias / grid + offset);
            circuit.append(OpKind::Reglu, Lin::constant(1.0), ramp, true, None)
        });
    }
    for wire in emitted {
        circuit.record_rounding_point(wire, RoundingPointKind::IntegerBit);
    }
    Some(bit)
}

/// The exact bit of D > 0, D >= 0 or D == 0, when every value of D is 0 or at
/// least 1 away from 0. `emit_ramp(sign, offset)` builds `relu(sign × D + offset)`:
/// D >= 0 is `relu(1 - relu(-D))`, D > 0 is `relu(1 - relu(1 - D))`, and
/// D == 0 is `relu(1 - relu(D) - relu(-D))`.
fn emit_integer_bit(
    circuit: &mut Circuit,
    test: SignTest,
    emitted: &mut Vec<Wire>,
    mut emit_ramp: impl FnMut(&mut Circuit, f64, f64) -> Wire,
) -> Wire {
    let ramps = if test == SignTest::Equal {
        vec![emit_ramp(circuit, 1.0, 0.0), emit_ramp(circuit, -1.0, 0.0)]
    } else {
        vec![emit_ramp(circuit, -1.0, if test == SignTest::Greater { 1.0 } else { 0.0 })]
    };
    let bit = circuit.reglu(1.0, Lin::new(ramps.iter().map(|&ramp| (ramp, -1.0)).collect(), 1.0));
    emitted.extend(ramps);
    emitted.push(bit);
    bit
}

/// Every nonzero FP32 difference, down to 2^-149, lifted to at least 1.
const FULL_LIFT_EXPONENT: i32 = 149;
/// Every nonzero FP32 difference lifted to at least 2^-45, which a final
/// rounding stage with scale 2^22 resolves.
const ROUNDING_STAGE_LIFT_EXPONENT: i32 = 104;

#[derive(Copy, Clone, Debug, PartialEq)]
struct CascadeStage {
    scale: f64,
    limit: f64,
}

/// Stages of `min(limit, scale × input)`. Multiplying by a power of two only
/// changes the exponent, and `min(limit, x) = x - relu(x - limit)` is exact
/// while x is at most 2^24 × limit.
fn plan_sign_cascade(difference_bound: f64, lift_exponent: i32) -> Vec<CascadeStage> {
    let bound_exponent = power_of_two_exponent_at_least(difference_bound.max(smallest_grid()));
    let mut stages = Vec::new();
    let mut limit_exponent = bound_exponent;
    let mut gain_exponent = 0;
    while stages.is_empty() || limit_exponent > 0 || gain_exponent < lift_exponent {
        let next_limit_exponent = (limit_exponent - 24).max(0);
        // A coefficient above 2^127 isn't an FP32 value.
        let room = (24 + next_limit_exponent - limit_exponent).min(127);
        let scale_exponent = room.min((lift_exponent - gain_exponent).max(0));
        stages.push(CascadeStage { scale: pow2(scale_exponent), limit: pow2(next_limit_exponent) });
        limit_exponent = next_limit_exponent;
        gain_exponent += scale_exponent;
    }
    stages
}

struct Cascade {
    difference: Wire,
    difference_sign: f64,
}

impl Cascade {
    /// After clamp stage k, the output is min(limit_k, scale_1 × ... × scale_k × relu(sign × d)).
    fn emit_clamp_stages(&self, circuit: &mut Circuit, sign: f64, stages: &[CascadeStage], emitted: &mut Vec<Wire>) -> Wire {
        let mut stage_output = self.difference;
        for (index, &CascadeStage { scale, limit }) in stages.iter().enumerate() {
            let (scaled_term, above_limit) = if index == 0 {
                // The first stage reads the signed difference as written, and keeps only its positive side.
                let gate: Vec<Term> = vec![(self.difference, sign * self.difference_sign * scale)];
                let scaled_positive = circuit.append(OpKind::Reglu, Lin::constant(1.0), Lin::new(gate.clone(), 0.0), true, None);
                let above_limit = circuit.append(OpKind::Reglu, Lin::constant(1.0), Lin::new(gate, -limit), true, None);
                emitted.push(scaled_positive);
                ((scaled_positive, 1.0), above_limit)
            } else {
                let above_limit = circuit.reglu(1.0, stage_output * scale - limit);
                ((stage_output, scale), above_limit)
            };
            // Its own operation: copied into the next stage, the regrouped partial sums would round.
            stage_output = circuit.round_affine(Lin::new(vec![scaled_term, (above_limit, -1.0)], 0.0));
            circuit.assume_range(stage_output, 0.0, limit);
            emitted.push(above_limit);
            emitted.push(stage_output);
        }
        stage_output
    }
}

fn emit_cascade_compare(
    circuit: &mut Circuit,
    terms: &[Term],
    bias: f64,
    difference_bound: f64,
    test: SignTest,
    end_with_boolean: bool,
) -> Wire {
    let mut emitted = Vec::new();
    // For finite FP32 operands, f(a - b) is 0 exactly when a == b, and otherwise
    // has the sign of a - b, since subnormal values fill the gap next to 0.
    let cascade = if bias == 0.0 && terms.len() == 1 {
        Cascade { difference: terms[0].0, difference_sign: terms[0].1 }
    } else {
        let difference = circuit.round_affine(Lin::new(terms.to_vec(), bias));
        emitted.push(difference);
        Cascade { difference, difference_sign: 1.0 }
    };
    let lift = if end_with_boolean { ROUNDING_STAGE_LIFT_EXPONENT } else { FULL_LIFT_EXPONENT };
    let stages = plan_sign_cascade(difference_bound, lift);
    let bit;
    if end_with_boolean {
        let score_scale = pow2(22);
        if test == SignTest::Greater {
            let stage_output = cascade.emit_clamp_stages(circuit, 1.0, &stages, &mut emitted);
            // The f32 value just below 0.5, so d = 0 rounds to 0.
            let just_below_half = 0.5 - pow2(-25);
            let score_above_zero = circuit.reglu(1.0, stage_output * score_scale + just_below_half);
            let score_above_one = circuit.reglu(1.0, stage_output * score_scale + (just_below_half - 1.0));
            let score = circuit.affine(score_above_zero - score_above_one);
            circuit.assume_range(score, 0.0, 1.0);
            emitted.push(score_above_zero);
            emitted.push(score_above_one);
            bit = circuit.boolean(score);
        } else {
            let mut gate = vec![(cascade.emit_clamp_stages(circuit, -1.0, &stages, &mut emitted), -score_scale)];
            if test == SignTest::Equal {
                gate.push((cascade.emit_clamp_stages(circuit, 1.0, &stages, &mut emitted), -score_scale));
            }
            let score = circuit.reglu(1.0, Lin::new(gate, 0.5));
            emitted.push(score);
            bit = circuit.boolean(score);
        }
        emitted.push(bit);
    } else {
        let final_scale = stages.last().unwrap().scale;
        let clamp_stages = &stages[..stages.len() - 1];
        if clamp_stages.is_empty() {
            // One stage lifts the signed difference itself, so D = scale × d is 0 or at least 1 away from 0.
            bit = emit_integer_bit(circuit, test, &mut emitted, |circuit, sign, offset| {
                let gate = Lin::new(vec![(cascade.difference, sign * cascade.difference_sign * final_scale)], offset);
                circuit.append(OpKind::Reglu, Lin::constant(1.0), gate, true, None)
            });
        } else if test == SignTest::Greater {
            // scale × stage is 0 when d <= 0, and at least 1 when d > 0.
            let stage = cascade.emit_clamp_stages(circuit, 1.0, clamp_stages, &mut emitted);
            let not_greater = circuit.reglu(1.0, 1.0 - stage * final_scale);
            bit = circuit.reglu(1.0, 1.0 - not_greater);
            emitted.push(not_greater);
            emitted.push(bit);
        } else {
            let mut gate = vec![(cascade.emit_clamp_stages(circuit, -1.0, clamp_stages, &mut emitted), -final_scale)];
            if test == SignTest::Equal {
                gate.push((cascade.emit_clamp_stages(circuit, 1.0, clamp_stages, &mut emitted), -final_scale));
            }
            bit = circuit.reglu(1.0, Lin::new(gate, 1.0));
            emitted.push(bit);
        }
    }
    for wire in emitted {
        circuit.record_rounding_point(wire, RoundingPointKind::SignCascade);
    }
    bit
}

/// `min(a, b)` as exactly one of the operands, e.g. f(0.3) for min(f(1000.37), f(0.3)).
/// Ties return `a`. Needs finite declared ranges, like [`emit_compare`].
pub fn emit_minimum_exact(circuit: &mut Circuit, a: Wire, b: Wire) -> Wire {
    let a_is_smaller = emit_compare(circuit, a, Relation::AtMost, b);
    let a_range = compare_operand(circuit, a).range;
    let b_range = compare_operand(circuit, b).range;
    emit_select_operand(circuit, a_is_smaller, b, a, a_range.min.min(b_range.min), a_range.max.min(b_range.max))
}

/// `max(a, b)` as exactly one of the operands. Ties return `a`.
pub fn emit_maximum_exact(circuit: &mut Circuit, a: Wire, b: Wire) -> Wire {
    let a_is_larger = emit_compare(circuit, a, Relation::AtLeast, b);
    let a_range = compare_operand(circuit, a).range;
    let b_range = compare_operand(circuit, b).range;
    emit_select_operand(circuit, a_is_larger, b, a, a_range.min.max(b_range.min), a_range.max.max(b_range.max))
}

/// `min(max(value, minimum), maximum)`, as exactly one of the operands.
pub fn emit_clamp_exact(circuit: &mut Circuit, value: Wire, minimum: Wire, maximum: Wire) -> Wire {
    let at_least_minimum = emit_maximum_exact(circuit, value, minimum);
    emit_minimum_exact(circuit, at_least_minimum, maximum)
}

/// `on × relu(bit) + off × relu(1 - bit)` for an exact bit: exactly the
/// selected operand, up to the sign of zero.
fn emit_select_operand(circuit: &mut Circuit, bit: Wire, off: Wire, on: Wire, minimum: f64, maximum: f64) -> Wire {
    let selected_on = circuit.reglu(on, bit);
    let selected_off = circuit.reglu(off, 1.0 - bit);
    let result = circuit.affine(selected_on + selected_off);
    circuit.assume_range(result, minimum, maximum);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cascade_plan_matches_handwired_comment() {
        // "a bound of 16,384 with liftExponent 149 gives scales 2^10, 2^24, 2^24, 2^24, 2^24, 2^24 and 2^19, all with limit 1"
        let scales: Vec<f64> = plan_sign_cascade(16_384.0, FULL_LIFT_EXPONENT).iter().map(|stage| stage.scale.log2()).collect();
        assert_eq!(scales, vec![10.0, 24.0, 24.0, 24.0, 24.0, 24.0, 19.0]);
        assert!(plan_sign_cascade(16_384.0, FULL_LIFT_EXPONENT).iter().all(|stage| stage.limit == 1.0));
    }
}
