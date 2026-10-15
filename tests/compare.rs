//! Comparisons: costs against handwired's published table, and every bit
//! against the native f32 comparison, through both evaluators.

mod common;

use common::{Rng, step_f32};
use handwired_rs::{
    Circuit, CompareOptions, Relation, RoundingPointKind, Wire, emit_compare_with, emit_maximum_exact, emit_minimum,
    emit_minimum_exact,
};

const RELATIONS: [Relation; 5] =
    [Relation::Greater, Relation::AtLeast, Relation::Less, Relation::AtMost, Relation::Equal];

fn two_operands(min: f64, max: f64, grid: Option<f64>) -> (Circuit, Wire, Wire) {
    let mut circuit = Circuit::new(2);
    let (a, b) = (circuit.source(0), circuit.source(1));
    for wire in [a, b] {
        circuit.assume_range(wire, min, max);
        if let Some(step) = grid {
            circuit.assume_grid(wire, step);
        }
    }
    (circuit, a, b)
}

/// (operations, layers) of one comparison.
fn cost(min: f64, max: f64, grid: Option<f64>, relation: Relation, end_with_boolean: bool) -> (usize, u32) {
    let (mut circuit, a, b) = two_operands(min, max, grid);
    let bit = emit_compare_with(&mut circuit, a, relation, b, CompareOptions { end_with_boolean });
    let live = circuit.compile(Some(&[bit])).live_operation_count();
    (live, circuit.depth(bit))
}

#[test]
fn costs_match_handwired_execution_model_table() {
    // docs/execution-model.md, "Costs, for operands within 0..16,384, or within ±8,192 on a 1/64 grid".
    let cascade = |relation, boolean| cost(0.0, 16_384.0, None, relation, boolean);
    let grid = |relation, boolean| cost(-8_192.0, 8_192.0, Some(1.0 / 64.0), relation, boolean);
    assert_eq!(cascade(Relation::Greater, false), (16, 15));
    assert_eq!(cascade(Relation::AtLeast, false), (15, 14));
    assert_eq!(cascade(Relation::Equal, false), (28, 14));
    assert_eq!(cascade(Relation::Greater, true), (15, 13));
    assert_eq!(cascade(Relation::AtLeast, true), (14, 13));
    assert_eq!(cascade(Relation::Equal, true), (25, 13));
    assert_eq!(grid(Relation::Greater, false), (2, 2));
    assert_eq!(grid(Relation::AtLeast, false), (2, 2));
    assert_eq!(grid(Relation::Equal, false), (3, 2));
    assert_eq!(grid(Relation::Greater, true), (1, 1));
    assert_eq!(grid(Relation::AtLeast, true), (1, 1));
    assert_eq!(grid(Relation::Equal, true), (3, 2));
}

#[test]
fn rounding_points_record_the_construction() {
    let (mut circuit, a, b) = two_operands(-8_192.0, 8_192.0, Some(1.0 / 64.0));
    emit_compare_with(&mut circuit, a, Relation::AtLeast, b, CompareOptions::default());
    assert!(circuit.rounding_points().iter().all(|&(_, kind)| kind == RoundingPointKind::IntegerBit));
    let (mut circuit, a, b) = two_operands(0.0, 16_384.0, None);
    emit_compare_with(&mut circuit, a, Relation::AtLeast, b, CompareOptions::default());
    assert!(circuit.rounding_points().iter().all(|&(_, kind)| kind == RoundingPointKind::SignCascade));
}

/// Pairs that stress a comparison: equal values, neighbors one ulp apart,
/// values next to zero, the range ends, and random values.
fn tricky_pairs(rng: &mut Rng, min: f32, max: f32, grid: Option<f32>, count: usize) -> Vec<(f32, f32)> {
    let snap = |value: f32| match grid {
        Some(step) => ((value / step).round() * step).clamp(min, max),
        None => value.clamp(min, max),
    };
    let neighbor = |value: f32, up: bool| match grid {
        Some(step) => snap(if up { value + step } else { value - step }),
        None => step_f32(value, up).clamp(min, max),
    };
    let mut pairs = vec![(min, max), (max, min), (min, min), (max, max)];
    if min <= 0.0 && max >= 0.0 {
        let tiny = grid.unwrap_or(f32::from_bits(1));
        pairs.extend([(0.0, 0.0), (0.0, -0.0), (0.0, tiny.min(max)), (tiny.min(max), 0.0), ((-tiny).max(min), 0.0)]);
    }
    while pairs.len() < count {
        let a = snap(rng.uniform(min as f64, max as f64) as f32);
        let b = match rng.below(4) {
            0 => a,
            1 => neighbor(a, true),
            2 => neighbor(a, false),
            _ => snap(rng.uniform(min as f64, max as f64) as f32),
        };
        pairs.push((a, b));
    }
    pairs
}

fn check_against_native(min: f64, max: f64, grid: Option<f64>, seed: u64) {
    // Declare ranges whose ends are f32 values, so every generated f32 lies inside them.
    let (min, max) = (min as f32 as f64, max as f32 as f64);
    let mut rng = Rng::new(seed);
    let pairs = tricky_pairs(&mut rng, min as f32, max as f32, grid.map(|step| step as f32), 3_000);
    for end_with_boolean in [false, true] {
        let (mut circuit, a, b) = two_operands(min, max, grid);
        let bits: Vec<Wire> = RELATIONS
            .iter()
            .map(|&relation| emit_compare_with(&mut circuit, a, relation, b, CompareOptions { end_with_boolean }))
            .collect();
        let program = circuit.compile(None);
        for &(x, y) in &pairs {
            let reference = circuit.evaluate_reference(&[x, y]).unwrap();
            let tensor = circuit.evaluate_tensor(&[x, y], &program).unwrap();
            circuit.validate_tensor_evaluation(&[x, y], &program, &tensor).unwrap();
            for (&relation, bit) in RELATIONS.iter().zip(&bits) {
                let expected = if relation.holds(x, y) { 1.0 } else { 0.0 };
                assert_eq!(
                    reference[bit.index()],
                    expected,
                    "{x:e} {} {y:e} (end_with_boolean {end_with_boolean})",
                    relation.symbol()
                );
            }
        }
    }
}

#[test]
fn sign_cascade_matches_native_comparison() {
    check_against_native(0.0, 16_384.0, None, 1);
    check_against_native(-1.0, 1.0, None, 2);
    check_against_native(-1e30, 1e30, None, 3);
    check_against_native(-1e-30, 1e-30, None, 4);
}

#[test]
fn integer_bits_match_native_comparison() {
    check_against_native(-8_192.0, 8_192.0, Some(1.0 / 64.0), 5);
    check_against_native(0.0, 1_000.0, Some(1.0), 6);
    // Too many grid steps for integer bits: falls back to the sign cascade.
    check_against_native(0.0, 262_144.0, Some(1.0 / 64.0), 7);
}

#[test]
fn constant_thresholds_fold_into_the_bias() {
    for grid in [None, Some(1.0)] {
        let mut circuit = Circuit::new(1);
        let x = circuit.source(0);
        circuit.assume_range(x, -100.0, 100.0);
        if let Some(step) = grid {
            circuit.assume_grid(x, step);
        }
        let three = circuit.constant(3.0);
        let bits: Vec<Wire> = RELATIONS
            .iter()
            .map(|&relation| emit_compare_with(&mut circuit, x, relation, three, CompareOptions::default()))
            .collect();
        for value in -100..=100 {
            let values = circuit.evaluate_reference(&[value as f32]).unwrap();
            for (&relation, bit) in RELATIONS.iter().zip(&bits) {
                assert_eq!(
                    values[bit.index()] == 1.0,
                    relation.holds(value as f32, 3.0),
                    "{value} {}",
                    relation.symbol()
                );
            }
        }
    }
}

#[test]
fn exact_minimum_returns_an_operand() {
    // docs/execution-model.md: min(f(1000.37), f(0.3)) is f(0.3) exactly, where emitMinimum gives 0.29998779.
    let (mut circuit, a, b) = two_operands(0.0, 16_384.0, None);
    let exact = emit_minimum_exact(&mut circuit, a, b);
    let rebuilt = emit_minimum(&mut circuit, a, b);
    let larger = emit_maximum_exact(&mut circuit, a, b);
    let values = circuit.evaluate_reference(&[1000.37, 0.3]).unwrap();
    assert_eq!(values[exact.index()], 0.3f32);
    assert_eq!(values[rebuilt.index()], 0.299_987_8_f32);
    assert_eq!(values[larger.index()], 1000.37f32);
}

#[test]
#[should_panic(expected = "finite declared ranges")]
fn open_ranges_are_refused() {
    let mut circuit = Circuit::new(2);
    let (a, b) = (circuit.source(0), circuit.source(1));
    emit_compare_with(&mut circuit, a, Relation::Greater, b, CompareOptions::default());
}
