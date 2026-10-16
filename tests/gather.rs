//! Gather lowering: a masked sum over a declared one-hot group compiles to a
//! read of the selected candidate. Ported from handwired's
//! tests/runtime/gather-lowering.test.ts. Every lowered program is checked
//! against the reference evaluator on every live wire.

mod common;

use common::Rng;
use handwired_rs::{BatchExecutor, Circuit, CompileOptions, TensorProgram, Wire, emit_maximum};

const DENSE: CompileOptions = CompileOptions { lower_gathers: false, recompute_depths: false };

/// Equal under `==`, which treats -0 and +0 as equal like handwired's checks, or both NaN.
fn same(left: f32, right: f32) -> bool {
    left == right || (left.is_nan() && right.is_nan())
}

/// Every live wire of the compiled program, through both executors, against the reference evaluator.
fn expect_exact(circuit: &Circuit, program: &TensorProgram, source: &[f32]) -> Vec<f32> {
    let reference = circuit.evaluate_reference(source).unwrap();
    let tensor = circuit.evaluate_tensor(source, program).unwrap();
    for wire in &program.wire_order {
        let (got, want) = (tensor[wire.index()], reference[wire.index()]);
        assert!(same(got, want), "wire {}: tensor {got:e}, reference {want:e}", wire.index());
    }
    let batched = BatchExecutor::new(program).run(source);
    for (root, got) in program.roots.iter().zip(batched) {
        let want = reference[root.index()];
        assert!(same(got, want), "root {}: batch {got:e}, reference {want:e}", root.index());
    }
    reference
}

fn stored_weights(program: &TensorProgram) -> usize {
    program.stored_weight_count()
}

fn gather_computes(program: &TensorProgram) -> usize {
    program.layers.iter().map(|layer| layer.gather_computes.len()).sum()
}

/// `slots` bits declared one-hot, `slots` nonnegative values, and one masked sum reading the selected value.
fn basic_read(slots: usize, bias: f64) -> (Circuit, Vec<Wire>, Vec<Wire>, Wire) {
    let mut circuit = Circuit::new(slots * 2);
    let bits: Vec<Wire> = (0..slots).map(|slot| circuit.source(slot)).collect();
    let values: Vec<Wire> = (0..slots).map(|slot| circuit.source(slots + slot)).collect();
    circuit.assume_one_hot(&bits, false);
    for &value in &values {
        circuit.assume_range(value, 0.0, 4096.0);
    }
    let read = masked_sum(&mut circuit, &bits, &values, bias);
    (circuit, bits, values, read)
}

fn masked_sum(circuit: &mut Circuit, bits: &[Wire], values: &[Wire], bias: f64) -> Wire {
    let terms: Vec<(Wire, f64)> =
        bits.iter().zip(values).map(|(&bit, &value)| (circuit.multiply(bit, value), 1.0)).collect();
    circuit.affine(handwired_rs::Lin::new(terms, bias))
}

fn basic_source(slots: usize, selected: Option<usize>) -> Vec<f32> {
    let mut source = vec![0.0; slots * 2];
    if let Some(selected) = selected {
        source[selected] = 1.0;
    }
    for slot in 0..slots {
        source[slots + slot] = 10.5 + slot as f32 * 3.25;
    }
    source
}

#[test]
fn a_masked_sum_lowers_to_one_gather_and_drops_its_products() {
    let slots = 8;
    let (circuit, _, _, read) = basic_read(slots, 0.0);
    let lowered = circuit.compile(Some(&[read]));
    let dense = circuit.compile_with(Some(&[read]), DENSE);
    assert_eq!(lowered.gather_read_count(), 1);
    assert_eq!(lowered.gather_slot_count, 1);
    // The dense form stores 2 weights per product and 1 per sum term; the gather row stores none, and the products
    // drop out of the program.
    assert_eq!(stored_weights(&dense), 3 * slots);
    assert_eq!(dense.gather_read_count(), 0);
    assert_eq!(stored_weights(&lowered), 0);
    assert_eq!(lowered.value_count, circuit.source_size() + 1);
    for selected in 0..slots {
        let source = basic_source(slots, Some(selected));
        let reference = expect_exact(&circuit, &lowered, &source);
        expect_exact(&circuit, &dense, &source);
        assert_eq!(reference[read.index()], source[slots + selected]);
    }
}

#[test]
fn a_bias_on_the_masked_sum_survives_lowering_exactly() {
    let (circuit, _, _, read) = basic_read(4, 0.1);
    let lowered = circuit.compile(Some(&[read]));
    assert_eq!(lowered.gather_read_count(), 1);
    for selected in 0..4 {
        let source = basic_source(4, Some(selected));
        let reference = expect_exact(&circuit, &lowered, &source);
        assert_eq!(reference[read.index()], 0.1f32 + source[4 + selected]);
    }
}

#[test]
fn signed_values_lower_through_the_mixed_sign_product() {
    // Declared after the products exist, the bits have unknown ranges, so `multiply` builds its mixed-sign form.
    let slots = 6;
    let mut circuit = Circuit::new(slots * 2);
    let bits: Vec<Wire> = (0..slots).map(|slot| circuit.source(slot)).collect();
    let values: Vec<Wire> = (0..slots).map(|slot| circuit.source(slots + slot)).collect();
    for &value in &values {
        circuit.assume_range(value, -4096.0, 4096.0);
    }
    let read = masked_sum(&mut circuit, &bits, &values, 0.0);
    circuit.assume_one_hot_allowing_readers(&bits, false);
    let lowered = circuit.compile(Some(&[read]));
    assert_eq!(lowered.gather_read_count(), 1);
    for selected in 0..slots {
        let mut source = vec![0.0; slots * 2];
        source[selected] = 1.0;
        for slot in 0..slots {
            source[slots + slot] = if slot % 2 == 0 { -1.0 } else { 1.0 } * (20.75 + slot as f32);
        }
        let reference = expect_exact(&circuit, &lowered, &source);
        assert_eq!(reference[read.index()], source[slots + selected]);
    }
}

#[test]
fn the_bit_in_the_gate_lowers_for_values_that_may_be_negative() {
    let slots = 5;
    let mut circuit = Circuit::new(slots * 2);
    let bits: Vec<Wire> = (0..slots).map(|slot| circuit.source(slot)).collect();
    let values: Vec<Wire> = (0..slots).map(|slot| circuit.source(slots + slot)).collect();
    circuit.assume_one_hot(&bits, false);
    for &value in &values {
        circuit.assume_range(value, -1e6, 1e6);
    }
    let read = masked_sum(&mut circuit, &bits, &values, 0.0);
    let lowered = circuit.compile(Some(&[read]));
    assert_eq!(lowered.gather_read_count(), 1);
    let mut source = vec![0.0; slots * 2];
    source[2] = 1.0;
    for slot in 0..slots {
        source[slots + slot] = -1000.5 + slot as f32 * 7.75;
    }
    let reference = expect_exact(&circuit, &lowered, &source);
    assert_eq!(reference[read.index()], source[slots + 2]);
}

#[test]
fn sums_that_are_not_pure_masked_reads_stay_dense_and_exact() {
    let slots = 4;
    type Mutation = fn(&mut Circuit, &mut Vec<(Wire, f64)>);
    let cases: [(&str, Mutation); 3] = [
        ("a scaled term", |_, terms| terms[1].1 = 2.0),
        ("a missing member", |_, terms| {
            terms.pop();
        }),
        ("an extra term", |circuit, terms| {
            let extra = circuit.source(8);
            terms.push((circuit.affine(extra + 5.0), 1.0));
        }),
    ];
    for (label, mutate) in cases {
        let mut circuit = Circuit::new(slots * 2 + 1);
        let bits: Vec<Wire> = (0..slots).map(|slot| circuit.source(slot)).collect();
        let values: Vec<Wire> = (0..slots).map(|slot| circuit.source(slots + slot)).collect();
        circuit.assume_one_hot(&bits, false);
        for &value in &values {
            circuit.assume_range(value, 0.0, 4096.0);
        }
        circuit.assume_range(circuit.source(8), 0.0, 4096.0);
        let mut terms: Vec<(Wire, f64)> =
            bits.iter().zip(&values).map(|(&bit, &value)| (circuit.multiply(bit, value), 1.0)).collect();
        mutate(&mut circuit, &mut terms);
        let read = circuit.affine(handwired_rs::Lin::new(terms, 0.0));
        let lowered = circuit.compile(Some(&[read]));
        assert_eq!(lowered.gather_read_count(), 0, "{label}");
        let mut source = vec![0.0; slots * 2 + 1];
        source[1] = 1.0;
        for slot in 0..slots {
            source[slots + slot] = 3.0 + slot as f32;
        }
        source[8] = 11.0;
        expect_exact(&circuit, &lowered, &source);
    }
}

#[test]
fn a_sum_over_two_groups_stays_dense() {
    let mut circuit = Circuit::new(8);
    circuit.assume_one_hot(&[circuit.source(0), circuit.source(1)], false);
    circuit.assume_one_hot(&[circuit.source(2), circuit.source(3)], false);
    for cell in 4..8 {
        circuit.assume_range(circuit.source(cell), 0.0, 100.0);
    }
    let bits: Vec<Wire> = (0..4).map(|cell| circuit.source(cell)).collect();
    let values: Vec<Wire> = (4..8).map(|cell| circuit.source(cell)).collect();
    let read = masked_sum(&mut circuit, &bits, &values, 0.0);
    let lowered = circuit.compile(Some(&[read]));
    assert_eq!(lowered.gather_read_count(), 0);
    expect_exact(&circuit, &lowered, &[1.0, 0.0, 0.0, 1.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn a_hand_written_bit_on_the_left_over_a_signed_value_is_rejected() {
    // f(bit × max(0, value)) reads the value only when the ReLU is the identity.
    let mut circuit = Circuit::new(4);
    let bits = [circuit.source(0), circuit.source(1)];
    let values = [circuit.source(2), circuit.source(3)];
    circuit.assume_one_hot(&bits, false);
    for value in values {
        circuit.assume_range(value, -100.0, 100.0);
    }
    let products: Vec<(Wire, f64)> =
        bits.iter().zip(values).map(|(&bit, value)| (circuit.reglu(bit, value), 1.0)).collect();
    let read = circuit.affine(handwired_rs::Lin::new(products, 0.0));
    let lowered = circuit.compile(Some(&[read]));
    assert_eq!(lowered.gather_read_count(), 0);
    expect_exact(&circuit, &lowered, &[0.0, 1.0, 7.0, -9.0]);
}

#[test]
fn a_one_member_group_lowers_its_biased_read_and_aliases_its_bare_read() {
    let mut circuit = Circuit::new(2);
    let (bit, value) = (circuit.source(0), circuit.source(1));
    circuit.assume_one_hot(&[bit], false);
    circuit.assume_range(value, 0.0, 100.0);
    let product = circuit.multiply(bit, value);
    let bare = circuit.affine(product);
    assert_eq!(bare, product);
    let biased = circuit.affine(product + 2.0);
    let lowered = circuit.compile(Some(&[bare, biased]));
    assert_eq!(lowered.gather_read_count(), 1);
    let reference = expect_exact(&circuit, &lowered, &[1.0, 42.0]);
    assert_eq!(reference[biased.index()], 44.0);
}

#[test]
fn reads_of_one_group_share_one_index_resolution() {
    let (slots, fields) = (8, 5);
    let mut circuit = Circuit::new(slots + slots * fields);
    let bits: Vec<Wire> = (0..slots).map(|slot| circuit.source(slot)).collect();
    circuit.assume_one_hot(&bits, false);
    let mut reads = Vec::new();
    for field in 0..fields {
        let values: Vec<Wire> = (0..slots).map(|slot| circuit.source(slots + field * slots + slot)).collect();
        for &value in &values {
            circuit.assume_range(value, 0.0, 1e6);
        }
        reads.push(masked_sum(&mut circuit, &bits, &values, 0.0));
    }
    let lowered = circuit.compile(Some(&reads));
    assert_eq!(lowered.gather_read_count(), fields);
    assert_eq!(gather_computes(&lowered), 1);
    assert_eq!(lowered.gather_slot_count, 1);
    let mut source = vec![0.0; circuit.source_size()];
    source[3] = 1.0;
    for (cell, value) in source.iter_mut().enumerate().skip(slots) {
        *value = cell as f32 * 1.5;
    }
    expect_exact(&circuit, &lowered, &source);
}

#[test]
fn identical_reads_deduplicate_to_one_gather() {
    let (mut circuit, bits, values, read) = basic_read(4, 0.0);
    let duplicate = masked_sum(&mut circuit, &bits, &values, 0.0);
    assert_eq!(duplicate, read);
    assert_eq!(circuit.compile(Some(&[read, duplicate])).gather_read_count(), 1);
}

#[test]
fn a_gather_can_read_another_gather() {
    let mut circuit = Circuit::new(11);
    let first_bits: Vec<Wire> = (0..4).map(|cell| circuit.source(cell)).collect();
    let first_values: Vec<Wire> = (4..8).map(|cell| circuit.source(cell)).collect();
    let second_bits = [circuit.source(8), circuit.source(9)];
    let plain = circuit.source(10);
    circuit.assume_one_hot(&first_bits, false);
    circuit.assume_one_hot(&second_bits, false);
    for &value in first_values.iter().chain([&plain]) {
        circuit.assume_range(value, 0.0, 1000.0);
    }
    let first = masked_sum(&mut circuit, &first_bits, &first_values, 0.0);
    let second = masked_sum(&mut circuit, &second_bits, &[first, plain], 0.0);
    let lowered = circuit.compile(Some(&[second]));
    assert_eq!(lowered.gather_read_count(), 2);
    for (first_slot, second_slot) in [(0, 0), (2, 0), (3, 1)] {
        let mut source = vec![0.0; 11];
        source[first_slot] = 1.0;
        source[8 + second_slot] = 1.0;
        for slot in 0..4 {
            source[4 + slot] = 100.5 + slot as f32;
        }
        source[10] = 999.0;
        let reference = expect_exact(&circuit, &lowered, &source);
        assert_eq!(reference[second.index()], if second_slot == 0 { source[4 + first_slot] } else { 999.0 });
    }
}

#[test]
fn a_group_that_allows_all_zeros_multiplies_by_its_mass() {
    let slots = 4;
    let mut circuit = Circuit::new(slots * 2);
    let bits: Vec<Wire> = (0..slots).map(|slot| circuit.source(slot)).collect();
    let values: Vec<Wire> = (0..slots).map(|slot| circuit.source(slots + slot)).collect();
    circuit.assume_one_hot(&bits, true);
    for &value in &values {
        circuit.assume_range(value, 0.0, 4096.0);
    }
    let read = masked_sum(&mut circuit, &bits, &values, 0.0);
    let lowered = circuit.compile(Some(&[read]));
    assert_eq!(lowered.gather_read_count(), 1);
    assert!(lowered.layers.iter().flat_map(|layer| &layer.gather_reads).all(|read| read.masked));
    for selected in [None, Some(0), Some(1), Some(2), Some(3)] {
        let source = basic_source(slots, selected);
        let reference = expect_exact(&circuit, &lowered, &source);
        assert_eq!(reference[read.index()], selected.map_or(0.0, |slot| source[slots + slot]));
    }
}

#[test]
fn candidates_that_may_be_infinite_are_refused_in_every_orientation() {
    // With an infinite unselected value, the dense sum is f(Infinity × 0) = NaN, which a gather would skip.
    for (label, bits_first, value_first, range) in [
        ("bit on the left", true, false, (0.0, f64::INFINITY)),
        ("bit in the gate", true, false, (f64::NEG_INFINITY, f64::INFINITY)),
        ("value first, bit in the gate", true, true, (f64::NEG_INFINITY, f64::INFINITY)),
        ("mixed-sign product", false, false, (f64::NEG_INFINITY, f64::INFINITY)),
    ] {
        let mut circuit = Circuit::new(4);
        let bits = [circuit.source(0), circuit.source(1)];
        let values = [circuit.source(2), circuit.source(3)];
        if bits_first {
            circuit.assume_one_hot(&bits, false);
        }
        for value in values {
            circuit.assume_range(value, range.0, range.1);
        }
        let read = if value_first {
            let first = circuit.multiply(values[0], bits[0]);
            let second = circuit.multiply(values[1], bits[1]);
            circuit.affine(first + second)
        } else {
            masked_sum(&mut circuit, &bits, &values, 0.0)
        };
        if !bits_first {
            circuit.assume_one_hot_allowing_readers(&bits, false);
        }
        let lowered = circuit.compile(Some(&[read]));
        assert_eq!(lowered.gather_read_count(), 0, "{label}");
        let reference = expect_exact(&circuit, &lowered, &[1.0, 0.0, 5.0, f32::INFINITY]);
        assert!(reference[read.index()].is_nan(), "{label}");
    }
}

#[test]
fn a_zero_filler_term_over_an_unbounded_value_blocks_lowering() {
    let build = |filler_maximum: f64| {
        let mut circuit = Circuit::new(5);
        let bits = [circuit.source(0), circuit.source(1)];
        let values = [circuit.source(2), circuit.source(3)];
        let filler = circuit.source(4);
        circuit.assume_one_hot(&bits, false);
        for value in values {
            circuit.assume_range(value, 0.0, 100.0);
        }
        circuit.assume_range(filler, 0.0, filler_maximum);
        let first = circuit.multiply(bits[0], values[0]);
        let second = circuit.multiply(bits[1], values[1]);
        // f(x × max(0, -bit)): a zero for an exact bit, but NaN for an infinite x.
        let zero = circuit.reglu(filler, -bits[0]);
        let read = circuit.affine(first + second + zero);
        (circuit, read)
    };
    let (circuit, read) = build(100.0);
    let lowered = circuit.compile(Some(&[read]));
    assert_eq!(lowered.gather_read_count(), 1);
    expect_exact(&circuit, &lowered, &[0.0, 1.0, 7.0, 9.0, 50.0]);

    let (circuit, read) = build(f64::INFINITY);
    let lowered = circuit.compile(Some(&[read]));
    assert_eq!(lowered.gather_read_count(), 0);
    let reference = expect_exact(&circuit, &lowered, &[1.0, 0.0, 7.0, 9.0, f32::INFINITY]);
    assert!(reference[read.index()].is_nan());
}

#[test]
fn gather_outputs_feed_later_operations_exactly() {
    let slots = 6;
    let (mut circuit, bits, values, read) = basic_read(slots, 0.0);
    let rotated: Vec<Wire> = (0..slots).map(|slot| values[(slot + 1) % slots]).collect();
    let other = masked_sum(&mut circuit, &bits, &rotated, 0.0);
    let largest = emit_maximum(&mut circuit, read, other);
    let product = circuit.multiply(read, other);
    let lowered = circuit.compile(Some(&[largest, product]));
    assert_eq!(lowered.gather_read_count(), 2);
    for selected in 0..slots {
        expect_exact(&circuit, &lowered, &basic_source(slots, Some(selected)));
    }
}

#[test]
fn a_violated_declaration_diverges_and_validation_catches_it() {
    let slots = 4;
    let (circuit, _, _, read) = basic_read(slots, 0.0);
    let lowered = circuit.compile(Some(&[read]));
    let mut source = basic_source(slots, Some(0));
    source[2] = 1.0;
    let reference = circuit.evaluate_reference(&source).unwrap();
    let tensor = circuit.evaluate_tensor(&source, &lowered).unwrap();
    assert_ne!(tensor[read.index()], reference[read.index()]);
    assert!(circuit.validate_one_hot_groups(&reference).unwrap_err().contains("one-hot group"));
    let fine = circuit.evaluate_reference(&basic_source(slots, Some(1))).unwrap();
    circuit.validate_one_hot_groups(&fine).unwrap();
}

#[test]
fn extreme_candidates_gather_bit_for_bit() {
    let extremes = [
        f32::MAX,
        -f32::MAX,
        f32::MIN_POSITIVE,
        -f32::from_bits(1),
        -0.0,
        0.0,
        16_777_216.0,
        16_777_217.0,
        -16_777_215.0,
        0.1,
    ];
    let slots = extremes.len();
    let mut circuit = Circuit::new(slots * 2);
    let bits: Vec<Wire> = (0..slots).map(|slot| circuit.source(slot)).collect();
    let values: Vec<Wire> = (0..slots).map(|slot| circuit.source(slots + slot)).collect();
    circuit.assume_one_hot(&bits, false);
    for &value in &values {
        circuit.assume_range(value, -f32::MAX as f64, f32::MAX as f64);
    }
    let read = masked_sum(&mut circuit, &bits, &values, 0.0);
    let lowered = circuit.compile(Some(&[read]));
    assert_eq!(lowered.gather_read_count(), 1);
    for selected in 0..slots {
        let mut source = vec![0.0; slots * 2];
        source[selected] = 1.0;
        source[slots..].copy_from_slice(&extremes);
        let reference = expect_exact(&circuit, &lowered, &source);
        assert_eq!(reference[read.index()], extremes[selected]);
    }
}

#[test]
fn recomputed_depths_move_operations_after_a_gather_earlier() {
    // Depths with no live operation are skipped, so the dropped products cost a layer only when another live
    // operation shares their depth, here `beside`. Recomputing depths puts the gather in that first layer.
    let slots = 6;
    let (mut circuit, _, values, read) = basic_read(slots, 0.0);
    let beside = circuit.round_affine(values[0] + 1.0);
    let after = circuit.round_affine(read * 2.0 + 1.0);
    let roots = [beside, after];
    let built = circuit.compile(Some(&roots));
    let recomputed =
        circuit.compile_with(Some(&roots), CompileOptions { recompute_depths: true, ..Default::default() });
    assert_eq!(built.layers.len(), 3);
    assert_eq!(recomputed.layers.len(), 2);
    for selected in 0..slots {
        let source = basic_source(slots, Some(selected));
        expect_exact(&circuit, &built, &source);
        expect_exact(&circuit, &recomputed, &source);
    }
}

/// Random circuits with one or two groups, declared before or after their reads, masked sums over nonnegative,
/// signed and extreme-range values (some of them earlier reads), and ordinary operations on top.
#[test]
fn seeded_circuits_with_gathers_match_the_reference() {
    let mut rng = Rng::new(0x6a7_3e5);
    let mut total_reads = 0;
    for _ in 0..400 {
        let group_sizes: Vec<usize> = (0..1 + rng.below(2)).map(|_| 1 + rng.below(9)).collect();
        let (nonnegative, signed, extreme) = (6, 6, 4);
        let control_size: usize = group_sizes.iter().sum();
        let mut circuit = Circuit::new(control_size + nonnegative + signed + extreme);
        let mut groups = Vec::new();
        let mut start = 0;
        let mut late = Vec::new();
        for &size in &group_sizes {
            let members: Vec<Wire> = (start..start + size).map(|cell| circuit.source(cell)).collect();
            let allow_all_zero = rng.below(3) == 0;
            if rng.below(2) == 0 {
                circuit.assume_one_hot(&members, allow_all_zero);
            } else {
                late.push((members.clone(), allow_all_zero));
            }
            groups.push((members, allow_all_zero));
            start += size;
        }
        let pool_start = control_size;
        for cell in 0..nonnegative {
            circuit.assume_range(circuit.source(pool_start + cell), 0.0, 1e6);
        }
        for cell in 0..signed {
            circuit.assume_range(circuit.source(pool_start + nonnegative + cell), -1e6, 1e6);
        }
        for cell in 0..extreme {
            circuit.assume_range(
                circuit.source(pool_start + nonnegative + signed + cell),
                -f32::MAX as f64,
                f32::MAX as f64,
            );
        }
        let mut values: Vec<Wire> = (pool_start..circuit.source_size()).map(|cell| circuit.source(cell)).collect();
        let mut roots = Vec::new();
        for _ in 0..1 + rng.below(4) {
            let (members, _) = &groups[rng.below(groups.len())];
            let candidates: Vec<Wire> = members.iter().map(|_| values[rng.below(values.len())]).collect();
            let bias = [0.0, 0.5, -3.0][rng.below(3)];
            let read = masked_sum(&mut circuit, members, &candidates, bias);
            // Some reads feed later reads and ordinary operations, when their range is finite.
            let range = circuit.range(read);
            if range.min.is_finite() && range.max.is_finite() && range.min.abs().max(range.max.abs()) < 1e7 {
                values.push(read);
                let next = circuit.round_affine(read * 0.5 - 2.0);
                roots.push(next);
            }
            roots.push(read);
        }
        for (members, allow_all_zero) in late {
            circuit.assume_one_hot_allowing_readers(&members, allow_all_zero);
        }
        let options = CompileOptions { recompute_depths: rng.below(2) == 0, ..Default::default() };
        let lowered = circuit.compile_with(Some(&roots), options);
        total_reads += lowered.gather_read_count();
        let dense = circuit.compile_with(Some(&roots), DENSE);
        for _ in 0..4 {
            let mut source = vec![0.0f32; circuit.source_size()];
            let mut start = 0;
            for (members, allow_all_zero) in &groups {
                let choices = members.len() + usize::from(*allow_all_zero);
                let choice = rng.below(choices);
                if choice < members.len() {
                    source[start + choice] = 1.0;
                }
                start += members.len();
            }
            for cell in 0..nonnegative {
                source[pool_start + cell] = rng.uniform(0.0, 1e6) as f32;
            }
            for cell in 0..signed {
                source[pool_start + nonnegative + cell] = rng.uniform(-1e6, 1e6) as f32;
            }
            let extremes = [f32::MAX, -f32::MAX, f32::MIN_POSITIVE, f32::from_bits(1), -0.0, 16_777_216.0, 2.5, -0.1];
            for cell in 0..extreme {
                source[pool_start + nonnegative + signed + cell] = extremes[rng.below(extremes.len())];
            }
            let reference = expect_exact(&circuit, &lowered, &source);
            expect_exact(&circuit, &dense, &source);
            circuit.validate_one_hot_groups(&reference).unwrap();
        }
    }
    assert!(total_reads > 400, "only {total_reads} reads lowered");
}
