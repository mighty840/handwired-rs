//! The compiled program reproduces every live value of the reference
//! evaluator, on seeded random circuits.

mod common;

use common::{Rng, SOURCES, random_circuit, random_source};
use handwired_rs::{BatchExecutor, Circuit, CompareOptions, LANES, Relation, TensorExecutor, Wire, emit_compare_with};

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

/// The batch executor gives every root bit for bit what the one-record executor gives, in full and partial
/// batches, on one thread and on several.
#[test]
fn batches_match_the_one_record_executor() {
    let mut rng = Rng::new(7);
    for circuit_number in 0..60 {
        let circuit = random_circuit(&mut rng, 60);
        let roots: Vec<Wire> = (SOURCES..circuit.width()).step_by(3).map(Wire::from_index).collect();
        let program = circuit.compile(Some(&roots));
        let records = [1, LANES, 2 * LANES + 5][circuit_number % 3];
        let sources: Vec<f32> = (0..records).flat_map(|_| random_source(&mut rng)).collect();
        let mut executor = TensorExecutor::new(&program);
        let expected: Vec<u32> =
            sources.chunks(SOURCES).flat_map(|source| executor.run(source).to_vec()).map(f32::to_bits).collect();
        let batched: Vec<u32> = BatchExecutor::new(&program).run(&sources).into_iter().map(f32::to_bits).collect();
        assert_eq!(batched, expected);
        let parallel: Vec<u32> =
            BatchExecutor::run_parallel(&program, &sources, Some(3)).into_iter().map(f32::to_bits).collect();
        assert_eq!(parallel, expected);
    }
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
