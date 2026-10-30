//! Gradient tapes reproduce the reference backward pass.

use handwired_rs::{
    BackpropOptions, BooleanGradient, Circuit, ComparisonGradient, Error, GradientTape, IndexGradient, Wire, handwired,
};

/// The balance example's fight: dice against thresholds, doubled hits on a 6, products of health and what's left.
#[handwired]
#[allow(clippy::too_many_arguments)]
fn skirmish(
    #[range(2, 20)] damage_a: f32,
    #[range(2, 20)] damage_b: f32,
    #[range(1, 200)] health_a: f32,
    #[range(1, 200)] health_b: f32,
    #[range(0, 2)] armor_a: f32,
    #[range(0, 2)] armor_b: f32,
    #[range(2, 6)] hits_on_a: u8,
    #[range(2, 6)] hits_on_b: u8,
    #[range(1, 6)] rolls_a: [u8; 4],
    #[range(1, 6)] rolls_b: [u8; 4],
) -> f32 {
    let mut taken_a: f32 = 0.0;
    let mut taken_b: f32 = 0.0;
    for volley in 0..4 {
        if rolls_a[volley] >= hits_on_a {
            let mut hit = damage_a - armor_b;
            if rolls_a[volley] == 6 {
                hit *= 2.0;
            }
            taken_b += hit;
        }
        if rolls_b[volley] >= hits_on_b {
            let mut hit = damage_b - armor_a;
            if rolls_b[volley] == 6 {
                hit *= 2.0;
            }
            taken_a += hit;
        }
    }
    (health_a - taken_a) * health_b - (health_b - taken_b) * health_a
}

/// A stats table read at a computed index, and a threshold on what it gives.
#[handwired]
fn table_rule(#[range(1, 20)] params: [f32; 6], #[range(0, 2)] faction: u8, #[range(0, 1)] stat: u8) -> f32 {
    let value = params[(faction * 2 + stat) as usize] * 3.0 - 1.0;
    if value > 20.0 { value * 0.5 } else { value.max(params[0]) }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn between(&mut self, low: u64, high: u64) -> u64 {
        low + self.next() % (high - low + 1)
    }
}

fn reference(circuit: &Circuit, source: &[f32], output: Wire, seed: f32, options: BackpropOptions) -> Vec<f32> {
    let values = circuit.evaluate_reference(source).unwrap();
    let mut seeds = vec![0.0; circuit.width()];
    seeds[output.index()] = seed;
    let mut gradients = circuit.backpropagate_reference(&values, &seeds, options).unwrap();
    gradients.truncate(circuit.source_size());
    gradients
}

/// Every option combination that doesn't reject these circuits.
fn options() -> Vec<BackpropOptions> {
    let mut all = Vec::new();
    for index_gradient in [IndexGradient::Zero, IndexGradient::BackwardDifference] {
        for comparison_gradient in [
            ComparisonGradient::Exact,
            ComparisonGradient::Surrogate { width: 1.5 },
            ComparisonGradient::RelativeSurrogate { fraction: 0.1 },
        ] {
            all.push(BackpropOptions {
                boolean_gradient: BooleanGradient::StraightThrough,
                index_gradient,
                comparison_gradient,
            });
        }
    }
    all
}

/// The tape's gradients against the reference's: bit for bit, except under a surrogate, whose coefficients the
/// tape holds as f32.
fn check(circuit: &Circuit, output: Wire, sources: &[Vec<f32>]) {
    for options in options() {
        let tape = circuit.gradient_tape(output, options).unwrap();
        assert!(tape.width <= circuit.width());
        let exact = options.comparison_gradient == ComparisonGradient::Exact;
        for (record, source) in sources.iter().enumerate() {
            let seed = 0.25 + record as f32 * 0.5;
            let expected = reference(circuit, source, output, seed, options);
            let actual = tape.backpropagate(source, seed);
            for (cell, (&expected, &actual)) in expected.iter().zip(&actual).enumerate() {
                if exact {
                    assert_eq!(expected.to_bits(), actual.to_bits(), "{options:?} record {record} cell {cell}");
                } else {
                    let tolerance = 1e-5 * expected.abs().max(1e-30);
                    assert!(
                        (expected - actual).abs() <= tolerance,
                        "{options:?} record {record} cell {cell}: {expected} vs {actual}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_tape_reproduces_the_reference_backward_pass_of_a_fight() {
    let compiled = skirmish_circuit();
    let mut rng = Rng(0x5EED_F00D);
    let records: Vec<_> = (0..64)
        .map(|_| {
            let mut rolls = || [0; 4].map(|_: u8| rng.between(1, 6) as u8);
            let (a, b) = (rolls(), rolls());
            (
                rng.between(2, 20) as f32,
                rng.between(2, 20) as f32,
                rng.between(1, 200) as f32,
                rng.between(1, 200) as f32,
                rng.between(0, 2) as f32,
                rng.between(0, 2) as f32,
                rng.between(2, 6) as u8,
                rng.between(2, 6) as u8,
                a,
                b,
            )
        })
        .collect();
    let encoded = compiled.encode_records(&records).unwrap();
    let cells = compiled.circuit.source_size();
    let sources: Vec<Vec<f32>> = encoded.sources().chunks(cells).map(<[f32]>::to_vec).collect();
    check(&compiled.circuit, compiled.outputs[0], &sources);
}

#[test]
fn a_tape_reproduces_the_reference_through_index_reads_and_maxima() {
    let compiled = table_rule_circuit();
    let mut rng = Rng(0x7AB1E);
    let records: Vec<_> = (0..64)
        .map(|_| ([0; 6].map(|_: u8| rng.between(1, 20) as f32), rng.between(0, 2) as u8, rng.between(0, 1) as u8))
        .collect();
    let encoded = compiled.encode_records(&records).unwrap();
    let cells = compiled.circuit.source_size();
    let sources: Vec<Vec<f32>> = encoded.sources().chunks(cells).map(<[f32]>::to_vec).collect();
    check(&compiled.circuit, compiled.outputs[0], &sources);
}

#[test]
fn a_tape_keeps_only_what_the_gradient_needs() {
    // Two outputs that share nothing: the tape of one leaves out the other's operations.
    let mut circuit = Circuit::new(2);
    let (a, b) = (circuit.source(0), circuit.source(1));
    circuit.assume_range(a, -4.0, 4.0);
    circuit.assume_range(b, -4.0, 4.0);
    let left = circuit.relu(a * 2.0 + 1.0);
    let right = circuit.multiply(b, b);
    let tape: GradientTape = circuit.gradient_tape(left, BackpropOptions::default()).unwrap();
    assert_eq!(tape.operation_count(), 1);
    assert_eq!(tape.backpropagate(&[1.0, 3.0], 1.0), vec![2.0, 0.0]);
    let tape = circuit.gradient_tape(right, BackpropOptions::default()).unwrap();
    assert_eq!(tape.backpropagate(&[1.0, 3.0], 1.0), vec![0.0, 6.0]);
}

#[test]
fn rejecting_rules_are_errors_when_the_tape_is_made() {
    let mut circuit = Circuit::new(1);
    let score = circuit.source(0);
    circuit.assume_range(score, 0.0, 1.0);
    let bit = circuit.boolean(score);
    let doubled = circuit.affine(bit * 2.0);
    assert!(matches!(circuit.gradient_tape(doubled, BackpropOptions::default()), Err(Error::BooleanGradient { .. })));
    let straight = BackpropOptions { boolean_gradient: BooleanGradient::StraightThrough, ..Default::default() };
    assert_eq!(circuit.gradient_tape(doubled, straight).unwrap().backpropagate(&[0.7], 1.0), vec![2.0]);
    let compiled = table_rule_circuit();
    assert!(matches!(compiled.circuit.gradient_tape(compiled.outputs[0], straight), Err(Error::IndexGradient { .. })));
}

#[test]
fn a_tape_reproduces_the_reference_through_booleans_and_both_ends_of_a_table() {
    // index = boolean(score) + 2 × step, which reads the table past both ends, clamped to its first and last entries.
    let mut circuit = Circuit::new(5);
    let (score, step) = (circuit.source(0), circuit.source(1));
    circuit.assume_range(score, 0.0, 1.0);
    circuit.assume_range(step, -1.0, 2.0);
    let table: Vec<Wire> = (2..5).map(|position| circuit.source(position)).collect();
    for &entry in &table {
        circuit.assume_range(entry, -50.0, 50.0);
    }
    let bit = circuit.boolean(score);
    let index = circuit.affine(bit + step * 2.0);
    let read = circuit.index_read(index, &table, handwired_rs::OutOfRange::Clamp);
    let output = circuit.multiply(read, step);
    let mut rng = Rng(0xB001);
    let sources: Vec<Vec<f32>> = (0..64)
        .map(|_| {
            let score = rng.between(0, 8) as f32 / 8.0;
            let step = rng.between(0, 6) as f32 * 0.5 - 1.0;
            vec![score, step, rng.between(0, 100) as f32 - 50.0, rng.between(0, 100) as f32 - 50.0, 7.0]
        })
        .collect();
    check(&circuit, output, &sources);
}
