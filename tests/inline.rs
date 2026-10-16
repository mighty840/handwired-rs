//! Inlining one circuit into another, ported from handwired's tests of
//! `inlineCircuit`, plus random circuits inlined into a larger circuit.

mod common;

use common::{Rng, SOURCES, random_circuit, random_source};
use handwired_rs::{Circuit, Lin, Wire};

#[test]
fn inlined_circuits_become_one_graph_and_share_matching_expressions() {
    let mut child = Circuit::new(2);
    let child_sum = child.affine(child.source(0) + child.source(1));
    let child_product = child.reglu(child_sum, child.source(0));
    let mut parent = Circuit::new(2);
    let existing_sum = parent.affine(parent.source(0) + parent.source(1));
    let map = parent.inline_circuit(&child, &[parent.source(0), parent.source(1)], false);
    assert_eq!(map[child_sum.index()], existing_sum);
    let output = map[child_product.index()];
    let program = parent.compile(Some(&[output]));
    assert_eq!(parent.evaluate_tensor(&[2.0, 3.0], &program).unwrap()[output.index()], 10.0);
}

#[test]
fn explicit_rounding_survives_consumers_duplicate_terms_and_inlining() {
    let mut circuit = Circuit::new(2);
    let (a, b) = (circuit.source(0), circuit.source(1));
    let sum = circuit.round_affine(a + b);
    let difference = circuit.affine(sum - a);
    let duplicate = circuit.round_affine(Lin::new(vec![(a, 1.0), (a, 1.0), (a, 1.0)], 0.0));
    let mut parent = Circuit::new(2);
    let map = parent.inline_circuit(&circuit, &[parent.source(0), parent.source(1)], false);
    let source = [16_777_216.0f32, 1.0];
    let values = circuit.evaluate_reference(&source).unwrap();
    assert_eq!(values[difference.index()], 0.0);
    assert_eq!(values[duplicate.index()], (source[0] + source[0]) + source[0]);
    let parent_values = parent.evaluate_reference(&source).unwrap();
    assert_eq!(parent_values[map[difference.index()].index()], 0.0);
    let program = parent.compile(None);
    assert_eq!(parent.evaluate_tensor(&source, &program).unwrap()[map[difference.index()].index()], 0.0);
}

#[test]
fn explicit_term_order_survives_a_permutation_of_the_cells() {
    let mut circuit = Circuit::new(3);
    let sum = circuit.round_affine(circuit.source(0) + circuit.source(1) + circuit.source(2));
    let mut parent = Circuit::new(3);
    let map = parent.inline_circuit(&circuit, &[parent.source(0), parent.source(2), parent.source(1)], false);
    let original = [16_777_216.0f32, 1.0, -16_777_216.0];
    let permuted = [16_777_216.0f32, -16_777_216.0, 1.0];
    assert_eq!(circuit.evaluate_reference(&original).unwrap()[sum.index()], 0.0);
    assert_eq!(parent.evaluate_reference(&permuted).unwrap()[map[sum.index()].index()], 0.0);
    let reordered = parent.round_affine(parent.source(0) + parent.source(1) + parent.source(2));
    assert_ne!(reordered, map[sum.index()]);
    assert_eq!(parent.evaluate_reference(&permuted).unwrap()[reordered.index()], 1.0);
}

#[test]
fn declarations_landing_on_read_wires_need_permission() {
    let mut child = Circuit::new(1);
    child.assume_range(child.source(0), 0.0, 1.0);
    let mut parent = Circuit::new(1);
    let cell = parent.source(0);
    parent.multiply(cell, cell);
    let mut refused = parent.clone();
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        refused.inline_circuit(&child, &[cell], false);
    }));
    assert!(panicked.is_err());
    parent.inline_circuit(&child, &[cell], true);
    assert_eq!(parent.range(cell).max, 1.0);
}

#[test]
fn an_exact_bit_lands_on_its_mapped_wire_only() {
    let build = |bit: bool| {
        let mut inner = Circuit::new(2);
        let (enabled, width) = (inner.source(0), inner.source(1));
        if bit {
            inner.assume_bit(enabled);
        } else {
            inner.assume_range(enabled, 0.0, 1.0);
        }
        inner.assume_range(width, 0.0, 100.0);
        inner.assume_grid(width, 1.0);
        inner.multiply(enabled, width);
        let mut outer = Circuit::new(3);
        for cell in 0..3 {
            outer.assume_range(outer.source(cell), 0.0, 100.0);
        }
        let map = outer.inline_circuit(&inner, &[outer.source(0), outer.source(1)], true);
        (outer, map)
    };
    let (with_bit, map) = build(true);
    let (without, _) = build(false);
    let bits: Vec<usize> =
        (0..with_bit.width()).filter(|&wire| with_bit.is_exact_bit(Wire::from_index(wire))).collect();
    assert_eq!(bits, [0]);
    assert!((0..without.width()).all(|wire| !without.is_exact_bit(Wire::from_index(wire))));
    // The declaration doesn't change the operations built.
    assert_eq!(with_bit.operations().len(), without.operations().len());
    let program = with_bit.compile(Some(&[*map.last().unwrap()]));
    assert_eq!(with_bit.evaluate_tensor(&[1.0, 42.0, 0.0], &program).unwrap()[map.last().unwrap().index()], 42.0);
}

#[test]
fn one_hot_groups_carry_over_and_still_lower() {
    let mut child = Circuit::new(4);
    let bits = [child.source(0), child.source(1)];
    let values = [child.source(2), child.source(3)];
    child.assume_one_hot(&bits, true);
    for value in values {
        child.assume_range(value, 0.0, 100.0);
    }
    let first = child.multiply(bits[0], values[0]);
    let second = child.multiply(bits[1], values[1]);
    let read = child.affine(first + second);
    let mut parent = Circuit::new(5);
    for cell in 1..5 {
        parent.assume_range(parent.source(cell), 0.0, if cell < 3 { 1.0 } else { 100.0 });
    }
    let mapping: Vec<Wire> = (1..5).map(|cell| parent.source(cell)).collect();
    let map = parent.inline_circuit(&child, &mapping, false);
    assert_eq!(parent.one_hot_groups().len(), 1);
    assert_eq!(parent.one_hot_groups()[0].members, [parent.source(1), parent.source(2)]);
    assert!(parent.one_hot_groups()[0].allow_all_zero);
    let output = map[read.index()];
    let program = parent.compile(Some(&[output]));
    assert_eq!(program.gather_read_count(), 1);
    let source = [9.0, 0.0, 1.0, 5.0, 7.0];
    let tensor = parent.evaluate_tensor(&source, &program).unwrap();
    assert_eq!(tensor[output.index()], 7.0);
    parent.validate_tensor_evaluation(&source, &program, &tensor).unwrap();
}

#[test]
fn random_circuits_inline_and_keep_every_value() {
    let mut rng = Rng::new(17);
    for _ in 0..100 {
        let child = random_circuit(&mut rng, 50);
        // The parent's cells: the child's, spread out between some of its own. The order is kept: a plain affine's
        // terms are summed in wire order, so a permutation could change its last bit, as in handwired.
        let mut parent = Circuit::new(2 * SOURCES + 1);
        let mapping: Vec<Wire> = (0..SOURCES).map(|cell| parent.source(2 * cell + 1)).collect();
        let map = parent.inline_circuit(&child, &mapping, false);
        for _ in 0..10 {
            let source = random_source(&mut rng);
            let mut parent_source = vec![0.0f32; 2 * SOURCES + 1];
            for (cell, &value) in source.iter().enumerate() {
                parent_source[2 * cell + 1] = value;
            }
            let child_values = child.evaluate_reference(&source).unwrap();
            let parent_values = parent.evaluate_reference(&parent_source).unwrap();
            for (wire, &mapped) in map.iter().enumerate() {
                let (want, got) = (child_values[wire], parent_values[mapped.index()]);
                assert!(want == got || (want.is_nan() && got.is_nan()), "wire {wire}: {want} vs {got}");
            }
            let roots: Vec<Wire> = map[SOURCES..].to_vec();
            let program = parent.compile(Some(&roots));
            let tensor = parent.evaluate_tensor(&parent_source, &program).unwrap();
            parent.validate_tensor_evaluation(&parent_source, &program, &tensor).unwrap();
        }
    }
}
