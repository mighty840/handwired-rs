//! Gradients of the intended real arithmetic.

use handwired_rs::{BackpropOptions, BooleanGradient, Circuit, Error, IndexGradient, OutOfRange, Wire};

fn gradients_of(circuit: &Circuit, source: &[f32], output: Wire, options: BackpropOptions) -> Result<Vec<f32>, Error> {
    let values = circuit.evaluate_reference(source).unwrap();
    let mut seed = vec![0.0; circuit.width()];
    seed[output.index()] = 1.0;
    circuit.backpropagate_reference(&values, &seed, options)
}

#[test]
fn products_and_selects_are_differentiated_as_a_whole() {
    // z = relu(2a - b) × a + (bit ? a : b), at a = 3, b = 1, bit = 1:
    // relu(5) × 3 + 3, so dz/da = 2 × 3 + 5 + 1 = 12, dz/db = -3, dz/dbit = a - b = 2.
    let mut circuit = Circuit::new(3);
    let (a, b, bit) = (circuit.source(0), circuit.source(1), circuit.source(2));
    circuit.assume_range(a, -10.0, 10.0);
    circuit.assume_range(b, -10.0, 10.0);
    circuit.assume_bit(bit);
    let hinge = circuit.relu(a * 2.0 - b);
    let product = circuit.multiply(hinge, a);
    let chosen = circuit.select(bit, b, a);
    let z = circuit.affine(product + chosen);
    let gradients = gradients_of(&circuit, &[3.0, 1.0, 1.0], z, BackpropOptions::default()).unwrap();
    assert_eq!(&gradients[..3], &[12.0, -3.0, 2.0]);
    // Past the hinge, relu(2a - b) contributes nothing.
    let gradients = gradients_of(&circuit, &[-3.0, 1.0, 0.0], z, BackpropOptions::default()).unwrap();
    assert_eq!(&gradients[..3], &[0.0, 1.0, -4.0]);
}

#[test]
fn boolean_atoms_need_an_explicit_gradient_rule() {
    let mut circuit = Circuit::new(1);
    let score = circuit.source(0);
    circuit.assume_range(score, 0.0, 1.0);
    let bit = circuit.boolean(score);
    let doubled = circuit.affine(bit * 2.0);
    assert!(matches!(
        gradients_of(&circuit, &[0.7], doubled, BackpropOptions::default()),
        Err(Error::BooleanGradient { .. })
    ));
    let options = BackpropOptions { boolean_gradient: BooleanGradient::StraightThrough, ..Default::default() };
    assert_eq!(gradients_of(&circuit, &[0.7], doubled, options).unwrap()[0], 2.0);
}

#[test]
fn index_reads_route_gradient_to_the_selected_candidate() {
    let mut circuit = Circuit::new(4);
    let index = circuit.source(0);
    circuit.assume_range(index, 0.0, 2.0);
    let table: Vec<Wire> = (1..4).map(|position| circuit.source(position)).collect();
    let read = circuit.index_read(index, &table, OutOfRange::Clamp);
    let source = [1.0, 10.0, 20.0, 40.0];
    assert!(matches!(
        gradients_of(&circuit, &source, read, BackpropOptions::default()),
        Err(Error::IndexGradient { .. })
    ));
    let zero = BackpropOptions { index_gradient: IndexGradient::Zero, ..Default::default() };
    assert_eq!(gradients_of(&circuit, &source, read, zero).unwrap(), vec![0.0, 0.0, 1.0, 0.0, 1.0]);
    // A backward difference gives the index read(1) - read(0).
    let difference = BackpropOptions { index_gradient: IndexGradient::BackwardDifference, ..Default::default() };
    assert_eq!(gradients_of(&circuit, &source, read, difference).unwrap()[0], 10.0);
}

/// A rule reading a stats table at a computed index, the way game rules read `params[faction * K + stat]`.
#[handwired_rs::handwired]
fn hit_damage(#[range(1, 20)] params: [f32; 6], #[range(0, 2)] faction: u8, #[range(0, 1)] stat: u8) -> f32 {
    let index = (faction * 2 + stat) as usize;
    params[index] * 3.0 - 1.0
}

#[test]
fn gradients_reach_the_table_entry_a_computed_index_reads() {
    use handwired_rs::dsl::Params;
    let compiled = hit_damage_circuit();
    compiled.verify(2_000).unwrap();
    let circuit = &compiled.circuit;
    let output = compiled.outputs[0];
    let options = BackpropOptions { index_gradient: IndexGradient::Zero, ..Default::default() };
    let params = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
    for faction in 0..3u8 {
        for stat in 0..2u8 {
            let mut source = Vec::new();
            (params, faction, stat).encode(&mut source);
            let gradients = gradients_of(circuit, &source, output, options).unwrap();
            // Cells 0..6 are the table: only the selected entry moves the output, by 3 per unit.
            let selected = usize::from(faction * 2 + stat);
            let want: Vec<f32> = (0..6).map(|cell| if cell == selected { 3.0 } else { 0.0 }).collect();
            assert_eq!(&gradients[..6], &want[..], "faction {faction}, stat {stat}");
        }
    }
    // Without a rule for the index, the default refuses a gradient at it rather than silently giving zero.
    let mut source = Vec::new();
    (params, 1u8, 0u8).encode(&mut source);
    assert!(matches!(
        gradients_of(circuit, &source, output, BackpropOptions::default()),
        Err(Error::IndexGradient { .. })
    ));
}

#[test]
fn batched_gradients_sum_the_per_record_backward_passes() {
    use handwired_rs::dsl::Params;
    let compiled = hit_damage_circuit();
    let options = BackpropOptions { index_gradient: IndexGradient::Zero, ..Default::default() };
    let records: Vec<([f32; 6], u8, u8)> = (0..997u32)
        .map(|n| {
            (std::array::from_fn(|cell| 1.0 + ((n as usize * 7 + cell * 3) % 19) as f32), (n % 3) as u8, (n % 2) as u8)
        })
        .collect();
    let weights: Vec<f32> = (0..records.len()).map(|n| [0.5, -1.0, 2.0, 0.0][n % 4]).collect();
    let batched = compiled.gradients(&records, 0, Some(&weights), options).unwrap();
    let mut expected = vec![0f64; 8];
    for (record, &weight) in records.iter().zip(&weights) {
        let mut source = Vec::new();
        record.encode(&mut source);
        let gradients = gradients_of(&compiled.circuit, &source, compiled.outputs[0], options).unwrap();
        for (sum, gradient) in expected.iter_mut().zip(&gradients[..8]) {
            *sum += (weight * gradient) as f64;
        }
    }
    assert_eq!(batched, expected);
    // Unweighted: each table entry's gradient is 3 × the number of records that read it.
    let unweighted = compiled.gradients(&records, 0, None, options).unwrap();
    assert_eq!(unweighted[..6].iter().sum::<f64>(), 3.0 * records.len() as f64);
}

#[test]
fn surrogate_gradients_go_through_comparisons_to_their_operands() {
    use handwired_rs::{CompareOptions, ComparisonGradient, Relation, emit_compare_with};
    let width = 0.5;
    let surrogate =
        BackpropOptions { comparison_gradient: ComparisonGradient::Surrogate { width }, ..Default::default() };
    let sigmoid = |d: f64| 1.0 / (1.0 + (-d / width).exp());
    // Integer bits on a grid, a sign cascade without one, and both ending in a boolean atom.
    for (grid, end_with_boolean) in [(Some(1.0), false), (None, false), (Some(1.0), true), (None, true)] {
        for relation in [Relation::Greater, Relation::AtLeast, Relation::Less, Relation::AtMost, Relation::Equal] {
            let mut circuit = Circuit::new(2);
            let (a, b) = (circuit.source(0), circuit.source(1));
            for wire in [a, b] {
                circuit.assume_range(wire, -100.0, 100.0);
                if let Some(step) = grid {
                    circuit.assume_grid(wire, step);
                }
            }
            let bit = emit_compare_with(&mut circuit, a, relation, b, CompareOptions { end_with_boolean });
            for (x, y) in [(3.0f32, 2.0f32), (2.0, 3.0), (5.0, 5.0), (-7.0, 4.0)] {
                let gradients = gradients_of(&circuit, &[x, y], bit, surrogate).unwrap();
                // The difference is oriented so the bit holds for d > 0 (or d >= 0, or d == 0).
                let d = match relation {
                    Relation::Less | Relation::AtMost => (y - x) as f64,
                    _ => (x - y) as f64,
                };
                let s = sigmoid(d);
                let slope = if relation == Relation::Equal {
                    4.0 * s * (1.0 - s) * (1.0 - 2.0 * s) / width
                } else {
                    s * (1.0 - s) / width
                };
                let (da, db) = match relation {
                    Relation::Less | Relation::AtMost => (-slope, slope),
                    _ => (slope, -slope),
                };
                let close = |got: f32, want: f64| {
                    assert!(
                        (got as f64 - want).abs() <= 1e-6 * want.abs().max(1.0),
                        "{relation:?} {grid:?} {end_with_boolean}: {got} vs {want}"
                    )
                };
                close(gradients[0], da);
                close(gradients[1], db);
            }
        }
    }
}

#[test]
fn surrogate_gradients_reach_through_max_and_thresholds_in_handwired_code() {
    use handwired_rs::ComparisonGradient;
    let compiled = threshold_rule_circuit();
    compiled.verify(2_000).unwrap();
    let surrogate =
        BackpropOptions { comparison_gradient: ComparisonGradient::Surrogate { width: 1.0 }, ..Default::default() };
    // Exactly, the rule is flat almost everywhere in `health`; the surrogate sees the threshold.
    let exact = compiled.gradients(&[(10.0, 9.5)], 0, None, BackpropOptions::default()).unwrap();
    let smooth = compiled.gradients(&[(10.0, 9.5)], 0, None, surrogate).unwrap();
    assert_eq!(exact[0], 0.0);
    assert!(smooth[0] < 0.0 && smooth[1] > 0.0, "{smooth:?}");
    // Far from the threshold, the surrogate fades out too.
    let far = compiled.gradients(&[(100.0, 1.0)], 0, None, surrogate).unwrap();
    assert!(far[0].abs() < 1e-30, "{far:?}");
}

/// 1 when a hit of `damage` takes `health` to 0 or below, else what's left as a fraction.
#[handwired_rs::handwired]
fn threshold_rule(#[range(0, 100)] health: f32, #[range(0, 100)] damage: f32) -> f32 {
    let left = (health - damage).max(0.0);
    if left <= 0.0 { 1.0 } else { 0.0 }
}

/// Whether a die roll meets a tunable threshold: the outcome depends on `threshold` only through a comparison.
#[handwired_rs::handwired]
fn meets(#[range(1, 7)] threshold: f32, #[range(1, 6)] roll: f32) -> f32 {
    if roll >= threshold { 1.0 } else { 0.0 }
}

#[test]
fn a_threshold_tunes_only_with_surrogate_gradients() {
    use handwired_rs::ComparisonGradient;
    let compiled = meets_circuit();
    compiled.verify(2_000).unwrap();
    // Tune the threshold until half the rolls meet it, by gradient descent on (rate - 0.5)².
    let tune = |options: BackpropOptions| -> f32 {
        let mut threshold = 1.5f32;
        for _ in 0..200 {
            let records: Vec<(f32, f32)> = (1..=6).map(|roll| (threshold, roll as f32)).collect();
            let rate = compiled.run_batch(&records).unwrap().iter().sum::<f32>() / 6.0;
            let weights = vec![2.0 * (rate - 0.5) / 6.0; 6];
            let gradient = compiled.gradients(&records, 0, Some(&weights), options).unwrap()[0];
            threshold = (threshold as f64 - 2.0 * gradient).clamp(1.0, 7.0) as f32;
        }
        threshold
    };
    // Exactly, the rate is flat in the threshold, so it never moves.
    assert_eq!(tune(BackpropOptions::default()), 1.5);
    // With a surrogate it moves until half the rolls (4, 5, 6) meet it.
    let surrogate =
        BackpropOptions { comparison_gradient: ComparisonGradient::Surrogate { width: 0.5 }, ..Default::default() };
    let tuned = tune(surrogate);
    assert!(tuned > 3.0 && tuned <= 4.0, "{tuned}");
}
