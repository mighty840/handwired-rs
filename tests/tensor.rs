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

#[test]
fn evenly_spaced_tables_store_a_base_and_a_stride() {
    use handwired_rs::{CandidatePositions, OutOfRange};
    // Ported from handwired's index-read test: other tables store positions shared by every read.
    let count = 1024;
    let mut circuit = Circuit::new(1 + count);
    let index = circuit.source(0);
    circuit.assume_range(index, 0.0, (count - 1) as f64);
    let values: Vec<Wire> = (1..=count).map(|position| circuit.source(position)).collect();
    for &wire in &values {
        circuit.assume_range(wire, -1e6, 1e6);
    }
    let reversed_values: Vec<Wire> = values.iter().rev().copied().collect();
    let every_other: Vec<Wire> = values.iter().copied().skip(1).step_by(2).collect();
    let mut rng = Rng::new(3);
    let mut shuffled = values.clone();
    for position in (1..shuffled.len()).rev() {
        shuffled.swap(position, rng.below(position + 1));
    }
    let contiguous = circuit.index_read(index, &values, OutOfRange::Clamp);
    let reversed = circuit.index_read(index, &reversed_values, OutOfRange::Clamp);
    let halves = circuit.index_read(index, &every_other, OutOfRange::Zero);
    let repeated = circuit.index_read(index, &[values[5]; 3], OutOfRange::Clamp);
    let scattered = circuit.index_read(index, &shuffled, OutOfRange::Clamp);
    let next = circuit.affine(index + 1.0);
    let scattered_next = circuit.index_read(next, &shuffled, OutOfRange::Zero);

    let program = circuit.compile(Some(&[contiguous]));
    assert_eq!(program.layers.len(), 1);
    assert_eq!(
        program.layers[0].index_reads[0].candidate_positions,
        CandidatePositions::Strided { base: 1, stride: 1, count: count as u32 }
    );
    let roots = [contiguous, reversed, halves, repeated, scattered, scattered_next];
    let program = circuit.compile(Some(&roots));
    let read_of = |wire: Wire| {
        let position = program.wire_order.iter().position(|&ordered| ordered == wire).unwrap() as u32;
        let mut offset = circuit.source_size() as u32;
        for layer in &program.layers {
            if let Some(read) = layer.index_reads.iter().find(|read| offset + read.row == position) {
                return read.candidate_positions.clone();
            }
            offset += layer.output_size as u32;
        }
        panic!("no index read for wire {}", wire.index());
    };
    assert_eq!(read_of(reversed), CandidatePositions::Strided { base: count as u32, stride: -1, count: count as u32 });
    assert_eq!(read_of(halves), CandidatePositions::Strided { base: 2, stride: 2, count: count as u32 / 2 });
    assert_eq!(read_of(repeated), CandidatePositions::Strided { base: 6, stride: 0, count: 3 });
    let (CandidatePositions::Listed(first), CandidatePositions::Listed(second)) =
        (read_of(scattered), read_of(scattered_next))
    else {
        panic!("a shuffled table is listed");
    };
    assert_eq!(first.len(), count);
    assert!(std::sync::Arc::ptr_eq(&first, &second), "reads of one table share its positions");

    // Both executors read every kind of table as the reference does, including past the ends.
    let mut executor = TensorExecutor::new(&program);
    for position in [0.0, 1.0, 511.0, 1022.0, 1023.0] {
        let mut source = vec![position];
        source.extend((0..count).map(|_| rng.below(2_000_001) as f32 - 1e6));
        let reference = circuit.evaluate_reference(&source).unwrap();
        let outputs = executor.run(&source).to_vec();
        for (&root, &output) in roots.iter().zip(&outputs) {
            assert_eq!(reference[root.index()].to_bits(), output.to_bits(), "index {position}, wire {}", root.index());
        }
    }
}
