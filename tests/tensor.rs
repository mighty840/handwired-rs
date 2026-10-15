//! The compiled program reproduces every live value of the reference
//! evaluator, on seeded random circuits.

mod common;

use common::Rng;
use handwired_rs::{emit_compare_with, Circuit, CompareOptions, Lin, OutOfRange, Relation, TensorExecutor, Wire};

const SOURCES: usize = 6;

/// A random circuit over 4 values within ±10, one bit, and one integer index within 0..7.
fn random_circuit(rng: &mut Rng, operations: usize) -> Circuit {
    let mut circuit = Circuit::new(SOURCES);
    for index in 0..4 {
        circuit.assume_range(circuit.source(index), -10.0, 10.0);
    }
    circuit.assume_bit(circuit.source(4));
    let index = circuit.source(5);
    circuit.assume_range(index, 0.0, 7.0);
    circuit.assume_grid(index, 1.0);
    let bit = circuit.source(4);
    let coefficients = [1.0, -1.0, 0.5, 2.0, -0.25, 0.1, 3.7, -1.3];
    let mut wires: Vec<Wire> = (0..4).map(|index| circuit.source(index)).collect();
    while circuit.operations().len() < operations {
        let bounded: Vec<Wire> = wires
            .iter()
            .copied()
            .filter(|&wire| {
                let range = circuit.range(wire);
                range.is_finite() && range.min.abs().max(range.max.abs()) < 1e4
            })
            .collect();
        let pick = |rng: &mut Rng| rng.pick(&bounded);
        let form = |rng: &mut Rng| {
            let count = 1 + rng.below(5);
            let terms = (0..count).map(|_| (pick(rng), rng.pick(&coefficients))).collect();
            Lin::new(terms, rng.pick(&coefficients))
        };
        let wire = match rng.below(7) {
            0 => circuit.affine(form(rng)),
            1 => circuit.round_affine(form(rng)),
            2 => circuit.reglu(form(rng), form(rng)),
            3 => circuit.multiply(pick(rng), pick(rng)),
            4 => circuit.select(bit, pick(rng), pick(rng)),
            5 => {
                let (a, b) = (pick(rng), pick(rng));
                let relation = rng.pick(&[Relation::Greater, Relation::AtLeast, Relation::Equal]);
                let end_with_boolean = rng.below(2) == 0;
                emit_compare_with(&mut circuit, a, relation, b, CompareOptions { end_with_boolean })
            }
            _ => {
                let table: Vec<Wire> = (0..1 + rng.below(8)).map(|_| pick(rng)).collect();
                let rule = rng.pick(&[OutOfRange::Clamp, OutOfRange::Zero]);
                circuit.index_read(index, &table, rule)
            }
        };
        wires.push(wire);
    }
    circuit
}

fn random_source(rng: &mut Rng) -> Vec<f32> {
    let mut source: Vec<f32> = (0..4).map(|_| rng.uniform(-10.0, 10.0) as f32).collect();
    source.push(rng.below(2) as f32);
    source.push(rng.below(8) as f32);
    source
}

#[test]
fn compiled_programs_match_the_reference_evaluator() {
    let mut rng = Rng::new(42);
    let mut total_layers = 0;
    for _ in 0..200 {
        let circuit = random_circuit(&mut rng, 60);
        let program = circuit.compile(None);
        total_layers += program.layers.len();
        for _ in 0..20 {
            let source = random_source(&mut rng);
            let tensor = circuit.evaluate_tensor(&source, &program).unwrap();
            circuit.validate_tensor_evaluation(&source, &program, &tensor).unwrap();
        }
    }
    assert!(total_layers > 200, "the random circuits should span several layers");
}

#[test]
fn compiling_for_roots_removes_dead_operations() {
    let mut rng = Rng::new(7);
    let circuit = random_circuit(&mut rng, 80);
    let root = Wire::from_index(circuit.width() - 1);
    let full = circuit.compile(None);
    let pruned = circuit.compile(Some(&[root]));
    assert!(pruned.live_operation_count() < full.live_operation_count());
    let mut executor = TensorExecutor::new(&pruned);
    for _ in 0..50 {
        let source = random_source(&mut rng);
        let reference = circuit.evaluate_reference(&source).unwrap();
        assert_eq!(executor.run(&source), &[reference[root.index()]]);
    }
}

#[test]
fn layers_follow_dependency_depth() {
    // (a > 4) ? 10 : 12 from handwired's README, written with a select.
    let mut circuit = Circuit::new(1);
    let a = circuit.source(0);
    circuit.assume_range(a, 0.0, 100.0);
    circuit.assume_grid(a, 1.0);
    let four = circuit.constant(4.0);
    let above_four = emit_compare_with(&mut circuit, a, Relation::Greater, four, CompareOptions::default());
    let (ten, twelve) = (circuit.constant(10.0), circuit.constant(12.0));
    let result = circuit.select(above_four, twelve, ten);
    let program = circuit.compile(Some(&[result]));
    assert_eq!(program.layers.len() as u32, circuit.depth(result));
    let mut executor = TensorExecutor::new(&program);
    for value in 0..=100 {
        assert_eq!(executor.run(&[value as f32]), &[if value > 4 { 10.0 } else { 12.0 }]);
    }
}
