//! `emit_argmax` and `emit_first_active`, ported from handwired's
//! numeric-ui-core and gather-lowering tests, with the costs its
//! docs/patterns.md states.

use handwired_rs::{BatchExecutor, Circuit, Lin, Wire, emit_argmax, emit_first_active, is_proven_bit};

/// `count` scores within 0..1 and a unit cell declared exactly 1, the last source cell.
fn scores_circuit(count: usize, extra: usize) -> (Circuit, Vec<Wire>, Wire) {
    let mut circuit = Circuit::new(count + 1 + extra);
    let scores: Vec<Wire> = (0..count).map(|item| circuit.source(item)).collect();
    for &score in &scores {
        circuit.assume_range(score, 0.0, 1.0);
    }
    let unit = circuit.source(count);
    circuit.assume_range(unit, 1.0, 1.0);
    (circuit, scores, unit)
}

#[test]
fn first_active_picks_the_first_hit_through_a_balanced_tree() {
    let count = 16;
    let mut circuit = Circuit::new(count + 1);
    for cell in 0..=count {
        circuit.assume_range(circuit.source(cell), 0.0, 1.0);
    }
    let gate = circuit.source(0);
    let candidates: Vec<Wire> = (1..=count).map(|item| circuit.source(item)).collect();
    let selected = emit_first_active(&mut circuit, gate, &candidates);
    let program = circuit.compile(Some(&selected));
    let all: Vec<usize> = (0..count).collect();
    for (active, hits, winner) in [
        (1.0, &[0][..], Some(0)),
        (1.0, &[2, 5], Some(2)),
        (1.0, &[1, 3, 14], Some(1)),
        (1.0, &[15], Some(15)),
        (0.0, &all, None),
    ] {
        let mut source = vec![0.0; count + 1];
        source[0] = active;
        for &hit in hits {
            source[hit + 1] = 1.0;
        }
        let values = circuit.evaluate_tensor(&source, &program).unwrap();
        let got: Vec<f32> = selected.iter().map(|wire| values[wire.index()]).collect();
        let want: Vec<f32> = (0..count).map(|item| if Some(item) == winner { 1.0 } else { 0.0 }).collect();
        assert_eq!(got, want, "hits {hits:?}");
    }
    assert!(program.layers.len() <= 5 * 4 + 1);
}

#[test]
fn argmax_picks_the_highest_score_through_a_balanced_tree() {
    let count = 7;
    let (mut circuit, scores, unit) = scores_circuit(count, 0);
    let (selected, score) = emit_argmax(&mut circuit, &scores, unit);
    let mut roots = selected.clone();
    roots.push(score);
    let program = circuit.compile(Some(&roots));
    for (scores, winner) in
        [([0.1, 0.7, 0.2, 0.4, 0.3, 0.6, 0.5], 1), ([0.8, 0.2, 0.8, 0.4, 0.8, 0.6, 0.5], 0), ([0.0; 7], 0)]
    {
        let mut source = scores.to_vec();
        source.push(1.0);
        let values = circuit.evaluate_tensor(&source, &program).unwrap();
        let got: Vec<f32> = selected.iter().map(|wire| values[wire.index()]).collect();
        let want: Vec<f32> = (0..count).map(|item| if item == winner { 1.0 } else { 0.0 }).collect();
        assert_eq!(got, want, "{scores:?}");
        assert_eq!(values[score.index()], scores[winner]);
    }
    assert!(program.layers.len() <= 6 * 3);
}

#[test]
fn argmax_breaks_exact_ties_toward_the_lower_index() {
    let (mut circuit, scores, unit) = scores_circuit(4, 0);
    let (selected, _) = emit_argmax(&mut circuit, &scores, unit);
    let program = circuit.compile(Some(&selected));
    for tie in [0.0, 0.25, 0.3, 0.5, 0.7, 1.0] {
        let values = circuit.evaluate_tensor(&[tie, tie, tie, tie, 1.0], &program).unwrap();
        let got: Vec<f32> = selected.iter().map(|wire| values[wire.index()]).collect();
        assert_eq!(got, [1.0, 0.0, 0.0, 0.0], "tie at {tie}");
    }
}

#[test]
fn argmax_keeps_the_left_score_within_its_tie_band() {
    // docs/execution-model.md: the band isn't transitive, so for [0, 0.5, 0.5 + 2^-24, 0.5 + 2^-23] the tree picks
    // index 1 where a loop keeping the strictly larger score picks index 3.
    let (mut circuit, scores, unit) = scores_circuit(4, 0);
    let (selected, _) = emit_argmax(&mut circuit, &scores, unit);
    let program = circuit.compile(Some(&selected));
    let half = 0.5f32;
    let source = [0.0, half, half + 2f32.powi(-24), half + 2f32.powi(-23), 1.0];
    let values = circuit.evaluate_tensor(&source, &program).unwrap();
    let got: Vec<f32> = selected.iter().map(|wire| values[wire.index()]).collect();
    assert_eq!(got, [0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn costs_match_handwireds_docs() {
    // docs/patterns.md: "At 64 candidates, emitFirstActive is 316 operations in 31 layers ... and emitArgmax is 437
    // in 33."
    let count = 64;
    let mut circuit = Circuit::new(count + 1);
    for cell in 0..=count {
        circuit.assume_range(circuit.source(cell), 0.0, 1.0);
    }
    let gate = circuit.source(0);
    let candidates: Vec<Wire> = (1..=count).map(|item| circuit.source(item)).collect();
    let selected = emit_first_active(&mut circuit, gate, &candidates);
    let program = circuit.compile(Some(&selected));
    assert_eq!((program.live_operation_count(), program.layers.len()), (316, 31));

    let (mut circuit, scores, unit) = scores_circuit(count, 0);
    let (selected, _) = emit_argmax(&mut circuit, &scores, unit);
    let program = circuit.compile(Some(&selected));
    assert_eq!((program.live_operation_count(), program.layers.len()), (437, 33));
}

#[test]
fn argmax_declares_its_group_and_first_active_only_for_proven_bits() {
    let (mut circuit, scores, unit) = scores_circuit(3, 1);
    let gate = circuit.source(4);
    circuit.assume_range(gate, 0.0, 1.0);
    let (selected, _) = emit_argmax(&mut circuit, &scores, unit);
    assert_eq!(circuit.one_hot_groups().len(), 1);
    assert_eq!(circuit.one_hot_groups()[0].members, selected);
    assert!(!circuit.one_hot_groups()[0].allow_all_zero);

    // A 0..1 range admits fractions, so this selection isn't provably one-hot.
    assert!(!is_proven_bit(&circuit, gate));
    emit_first_active(&mut circuit, gate, &scores);
    assert_eq!(circuit.one_hot_groups().len(), 1);

    // Boolean atoms are exact bits and the unit's range pins it to 1.
    let bits: Vec<Wire> = scores.iter().map(|&score| circuit.boolean(score)).collect();
    let selected = emit_first_active(&mut circuit, unit, &bits);
    assert_eq!(circuit.one_hot_groups().len(), 2);
    assert_eq!(circuit.one_hot_groups()[1].members, selected);
    assert!(circuit.one_hot_groups()[1].allow_all_zero);

    let values = circuit.evaluate_reference(&[0.5, 0.75, 0.25, 1.0, 0.0]).unwrap();
    circuit.validate_one_hot_groups(&values).unwrap();
}

#[test]
fn fractional_first_active_selections_never_lower_a_masked_sum() {
    // Candidates [0.5, 0] route 0.25 to each output; a group declared anyway would make the gather return 50 where
    // the sum is 75.
    let mut circuit = Circuit::new(5);
    let gate = circuit.source(0);
    let candidates = [circuit.source(1), circuit.source(2)];
    let values = [circuit.source(3), circuit.source(4)];
    for wire in [gate, candidates[0], candidates[1]] {
        circuit.assume_range(wire, 0.0, 1.0);
    }
    for value in values {
        circuit.assume_range(value, 0.0, 1000.0);
    }
    let selected = emit_first_active(&mut circuit, gate, &candidates);
    let first = circuit.multiply(selected[0], values[0]);
    let second = circuit.multiply(selected[1], values[1]);
    let read = circuit.affine(first + second);
    assert!(circuit.one_hot_groups().is_empty());
    let program = circuit.compile(Some(&[read]));
    assert_eq!(program.gather_read_count(), 0);
    let source = [1.0, 0.5, 0.0, 100.0, 200.0];
    let tensor = circuit.evaluate_tensor(&source, &program).unwrap();
    assert_eq!(tensor[read.index()], 75.0);
    circuit.validate_tensor_evaluation(&source, &program, &tensor).unwrap();
}

#[test]
fn first_active_accepts_candidates_that_fold_to_one_constant() {
    let mut circuit = Circuit::new(1);
    let unit = circuit.source(0);
    circuit.assume_range(unit, 1.0, 1.0);
    let zero = circuit.constant(0.0);
    let selected = emit_first_active(&mut circuit, unit, &[zero, zero]);
    assert_eq!(selected.len(), 2);
    assert_eq!(circuit.one_hot_groups().len(), 1);
    let values = circuit.evaluate_reference(&[1.0]).unwrap();
    assert_eq!([values[selected[0].index()], values[selected[1].index()]], [0.0, 0.0]);
    circuit.validate_one_hot_groups(&values).unwrap();
}

#[test]
fn a_masked_read_of_an_argmax_selection_lowers_to_one_gather() {
    // Some selection wires are two-term affines that readers copy in, so recognition has to recover them through
    // the deduplication map.
    let (mut circuit, scores, unit) = scores_circuit(3, 3);
    let values: Vec<Wire> = (4..7).map(|cell| circuit.source(cell)).collect();
    for &value in &values {
        circuit.assume_range(value, 0.0, 4096.0);
    }
    let (selected, _) = emit_argmax(&mut circuit, &scores, unit);
    let terms: Vec<(Wire, f64)> =
        selected.iter().zip(&values).map(|(&bit, &value)| (circuit.multiply(bit, value), 1.0)).collect();
    let read = circuit.affine(Lin::new(terms, 0.0));
    let mut roots = vec![read];
    roots.extend(&selected);
    let program = circuit.compile(Some(&roots));
    assert_eq!(program.gather_read_count(), 1);
    for scores in [[0.2, 0.9, 0.5], [0.9, 0.2, 0.5], [0.5, 0.5, 0.5], [0.1, 0.2, 0.9]] {
        let mut source = scores.to_vec();
        source.extend([1.0, 17.5, 23.25, 31.125]);
        let reference = circuit.evaluate_reference(&source).unwrap();
        let tensor = circuit.evaluate_tensor(&source, &program).unwrap();
        circuit.validate_tensor_evaluation(&source, &program, &tensor).unwrap();
        let winner = selected.iter().position(|bit| reference[bit.index()] == 1.0).unwrap();
        assert_eq!(reference[read.index()], source[4 + winner]);
        assert_eq!(BatchExecutor::new(&program).run(&source)[0], source[4 + winner]);
    }
}
