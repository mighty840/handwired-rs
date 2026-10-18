//! The format's contract: a program exported by handwired-rs and run here computes every value bit for bit as
//! handwired-rs's `TensorExecutor` does, and damaged bytes are refused, never a panic.

use handwired_rs::dsl::HwType;
use handwired_rs::{
    Circuit, OutOfRange, TensorExecutor, TensorProgram, Wire, emit_argmax, emit_first_active, handwired,
};
use handwired_rs_run::{Error, Program};

/// xorshift64*.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Input cells within the declared ranges: each range's ends, then random values, whole steps of the cell's grid
/// where it has one.
fn records(program: &TensorProgram, count: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng(seed | 1);
    let sources = 0..program.source_size;
    let mut records = vec![
        sources.clone().map(|cell| program.ranges[cell].min as f32).collect(),
        sources.clone().map(|cell| program.ranges[cell].max as f32).collect(),
    ];
    for _ in 0..count {
        records.push(
            sources
                .clone()
                .map(|cell| {
                    let range = program.ranges[cell];
                    let value = range.min + rng.unit() * (range.max - range.min);
                    let value = match program.grids[cell] {
                        Some(step) => ((value / step).floor() * step).clamp(range.min, range.max),
                        None => value,
                    };
                    // Within the range as an f32 too.
                    let cell_value = value as f32;
                    if (cell_value as f64) < range.min || (cell_value as f64) > range.max {
                        range.min as f32
                    } else {
                        cell_value
                    }
                })
                .collect(),
        );
    }
    records
}

/// A run with scratch and outputs of its own (`Program::evaluate` needs the `alloc` feature).
fn evaluate(program: &Program, inputs: &[f32]) -> Result<Vec<f32>, Error> {
    let mut scratch = vec![0.0; program.scratch_len()];
    let mut outputs = vec![0.0; program.output_count()];
    program.run(inputs, &mut scratch, &mut outputs)?;
    Ok(outputs)
}

/// Export `program`, run it here on `count` records, and compare every value with `TensorExecutor`'s bits.
fn roundtrip(name: &str, program: &TensorProgram, count: usize) -> Vec<u8> {
    let bytes = program.to_bytes(name);
    let parsed = Program::parse(&bytes).unwrap_or_else(|error| panic!("{name}: {error}"));
    assert_eq!(parsed.description(), name);
    assert_eq!((parsed.input_count(), parsed.output_count()), (program.source_size, program.roots.len()));
    assert_eq!((parsed.layer_count(), parsed.value_count()), (program.layers.len(), program.value_count));
    let mut executor = TensorExecutor::new(program);
    let mut scratch = vec![0.0f32; parsed.scratch_len()];
    let mut outputs = vec![0.0f32; parsed.output_count()];
    for record in records(program, count, name.len() as u64 * 7919) {
        let expected: Vec<u32> = executor.run(&record).iter().map(|value| value.to_bits()).collect();
        parsed.run(&record, &mut scratch, &mut outputs).unwrap_or_else(|error| panic!("{name}: {error}"));
        let got: Vec<u32> = outputs.iter().map(|value| value.to_bits()).collect();
        assert_eq!(got, expected, "{name}: outputs for {record:?}");
        for (position, (&got, &want)) in scratch.iter().zip(&executor.packed_values).enumerate() {
            assert_eq!(got.to_bits(), want.to_bits(), "{name}: value {position} for {record:?}: {got} vs {want}");
        }
    }
    bytes
}

#[handwired]
fn bits(a: u16, b: u16, #[range(0, 7)] s: u8, c: i16) -> (u16, u16, u32, i16) {
    let x = (a ^ b) >> s;
    (x | (a & 0x0F0F), a.count_ones() as u16 + (b >> 3), ((a as u32) << 4) ^ (b as u32), c >> 3)
}

#[handwired]
#[allow(clippy::needless_range_loop)]
fn wide_hash(bytes: [u8; 8], #[range(0, 3)] amount: u8) -> ([u8; 4], [u8; 8], bool) {
    let mut hash = 0x811C_9DC5u32;
    let mut word = 0u64;
    for i in 0..8 {
        if i < 2 {
            hash ^= bytes[i] as u32;
            hash = hash.wrapping_mul(0x0100_0193);
        }
        word |= (bytes[i] as u64) << (8 * i);
    }
    let mixed = (word.rotate_left(17) ^ (word >> amount)).wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut out = [0u8; 8];
    for i in 0..8 {
        out[i] = (mixed >> (8 * i)) as u8;
    }
    ([hash as u8, (hash >> 8) as u8, (hash >> 16) as u8, (hash >> 24) as u8], out, mixed > word)
}

/// Pair tables (index reads) and long division.
#[handwired]
fn divide(
    #[range(0, 200)] pot: u16,
    #[range(1, 6)] players: u16,
    #[range(-60_000, 60_000)] a: i32,
    #[range(-300, 300)] b: i32,
) -> (u16, u16, i32) {
    let quotient = if b == 0 { 0 } else { a / b };
    (pot / players, pot % players, quotient)
}

/// What a unit does.
#[derive(Clone, Copy, Debug, PartialEq, HwType)]
enum Action {
    Wait,
    Move {
        #[range(-1, 1)]
        dx: i8,
    },
    Attack(#[range(1, 6)] u8),
}

/// Enums with data, Options, and reads at a computed index.
#[handwired]
#[allow(clippy::manual_map)]
fn plan(
    actions: [Action; 4],
    #[range(0, 3)] at: usize,
    #[range(0, 50)] hp: u8,
    bonus: Option<u8>,
) -> (Action, Option<u8>, bool) {
    let current = actions[at];
    let next = match current {
        Action::Attack(_) if hp < 10 => Action::Move { dx: -1 },
        Action::Attack(damage) => Action::Attack(damage.min(5) + 1),
        other => other,
    };
    let total = match bonus {
        Some(b) => Some(b.saturating_add(hp)),
        None => None,
    };
    (next, total, current == Action::Wait)
}

/// Floats: products, sums, comparisons and a clamp.
#[handwired]
fn physics(#[range(-10.0, 10.0)] x: f32, #[range(-4.0, 4.0)] v: f32, #[range(0.0, 1.0)] drag: f32) -> (f32, f32, bool) {
    let next_v = (v - v * drag * 0.5).clamp(-3.0, 3.0);
    let next_x = x + next_v * 0.25;
    (next_x, next_v, next_x > 0.0)
}

#[test]
fn circuits_from_functions_run_bit_for_bit() {
    roundtrip("bits", &bits_circuit().program, 300);
    roundtrip("wide_hash", &wide_hash_circuit().program, 100);
    roundtrip("divide", &divide_circuit().program, 300);
    roundtrip("plan", &plan_circuit().program, 300);
    roundtrip("physics", &physics_circuit().program, 500);
}

/// Gathers: argmax over scores selects a value, the first active bit (which allows none) a masked one.
fn gathers() -> TensorProgram {
    let size = 6;
    let mut circuit = Circuit::new(3 * size + 1);
    let unit = circuit.source(3 * size);
    circuit.assume_range(unit, 1.0, 1.0);
    let scores: Vec<Wire> = (0..size).map(|cell| circuit.source(cell)).collect();
    let flags: Vec<Wire> = (0..size).map(|cell| circuit.source(size + cell)).collect();
    let values: Vec<Wire> = (0..size).map(|cell| circuit.source(2 * size + cell)).collect();
    for cell in 0..size {
        circuit.assume_range(scores[cell], 0.0, 1.0);
        circuit.assume_range(flags[cell], 0.0, 1.0);
        circuit.assume_range(values[cell], -1000.0, 1000.0);
    }
    let picked = emit_argmax(&mut circuit, &scores, unit).0;
    let bits: Vec<Wire> = flags.iter().map(|&flag| circuit.boolean(flag)).collect();
    let first = emit_first_active(&mut circuit, unit, &bits);
    let mut sum = |selectors: &[Wire]| {
        let terms: Vec<(Wire, f64)> =
            selectors.iter().zip(&values).map(|(&bit, &value)| (circuit.multiply(bit, value), 1.0)).collect();
        circuit.affine(handwired_rs::Lin::new(terms, 0.5))
    };
    let (a, b) = (sum(&picked), sum(&first));
    let total = circuit.affine(handwired_rs::Lin::new(vec![(a, 1.0), (b, -2.0)], 0.0));
    let program = circuit.compile(Some(&[a, b, total]));
    assert!(program.gather_read_count() >= 2, "{} gather reads", program.gather_read_count());
    program
}

#[test]
fn gathers_run_bit_for_bit() {
    roundtrip("gathers", &gathers(), 500);
}

/// Table reads at fractional indices and past both ends, under each rule.
fn table_reads() -> TensorProgram {
    let size = 5;
    let mut circuit = Circuit::new(size + 1);
    let candidates: Vec<Wire> = (0..size).map(|cell| circuit.source(cell)).collect();
    for &candidate in &candidates {
        circuit.assume_range(candidate, -50.0, 50.0);
    }
    let index = circuit.source(size);
    circuit.assume_range(index, -3.0, size as f64 + 3.0);
    let mut roots = Vec::new();
    for (below, above) in [
        (OutOfRange::Zero, OutOfRange::Zero),
        (OutOfRange::Clamp, OutOfRange::Clamp),
        (OutOfRange::Zero, OutOfRange::Clamp),
        (OutOfRange::Clamp, OutOfRange::Zero),
    ] {
        let form = handwired_rs::Lin::new(vec![(index, 1.0)], 0.0);
        roots.push(circuit.index_read_affine(form, &candidates, below, above, None));
    }
    circuit.compile(Some(&roots))
}

#[test]
fn table_reads_past_their_ends_run_bit_for_bit() {
    let program = table_reads();
    roundtrip("table_reads", &program, 300);
    // The index exactly at, just past and just before each end.
    let bytes = program.to_bytes("table_reads");
    let parsed = Program::parse(&bytes).unwrap();
    let mut executor = TensorExecutor::new(&program);
    for index in [-3.0, -1.0, -0.5, -0.0, 0.0, 0.25, 3.999, 4.0, 4.5, 4.999_999_5, 5.0, 5.5, 6.0, 8.0] {
        let mut record = vec![11.0, -12.0, 13.5, -14.0, 15.0];
        record.push(index);
        let expected: Vec<u32> = executor.run(&record).iter().map(|value| value.to_bits()).collect();
        let got: Vec<u32> = evaluate(&parsed, &record).unwrap().iter().map(|value| value.to_bits()).collect();
        assert_eq!(got, expected, "index {index}");
    }
}

#[test]
fn inputs_outside_their_ranges_and_wrong_lengths_are_refused() {
    let program = bits_circuit().program;
    let bytes = program.to_bytes("bits");
    let parsed = Program::parse(&bytes).unwrap();
    let mut scratch = vec![0.0; parsed.scratch_len()];
    let mut outputs = vec![0.0; parsed.output_count()];
    assert_eq!(parsed.input_range(2), Some((0.0, 7.0)));
    let error = parsed.run(&[1.0, 2.0, 8.0, 0.0], &mut scratch, &mut outputs).unwrap_err();
    assert_eq!(error, Error::InputOutOfRange { index: 2, value: 8.0 });
    assert!(matches!(
        parsed.run(&[1.0, 2.0, f32::NAN, 0.0], &mut scratch, &mut outputs),
        Err(Error::InputOutOfRange { .. })
    ));
    assert!(matches!(parsed.run(&[1.0, 2.0], &mut scratch, &mut outputs), Err(Error::InputLength { .. })));
    assert!(matches!(
        parsed.run(&[1.0, 2.0, 3.0, 0.0], &mut scratch[1..], &mut outputs),
        Err(Error::ScratchTooSmall { .. })
    ));
    assert!(matches!(
        parsed.run(&[1.0, 2.0, 3.0, 0.0], &mut scratch, &mut outputs[1..]),
        Err(Error::OutputLength { .. })
    ));
}

#[test]
fn compiled_functions_export_their_signature() {
    let bytes = divide_circuit().export("divide");
    let program = Program::parse(&bytes).unwrap();
    assert!(program.description().starts_with("divide\nparameters: (u16, u16, i32, i32)\nresult: (u16, u16, i32)"));
    assert_eq!(evaluate(&program, &[200.0, 6.0, -60_000.0, 7.0]).unwrap(), vec![33.0, 2.0, -8571.0]);
}

#[test]
fn damaged_bytes_are_refused_without_a_panic() {
    let bytes = roundtrip("plan", &plan_circuit().program, 1);
    // Every truncation.
    for length in 0..bytes.len() {
        assert!(Program::parse(&bytes[..length]).is_err(), "{length} bytes parsed");
    }
    assert_eq!(Program::parse(b"not a program at all").unwrap_err(), Error::NotAProgram);
    let mut newer = bytes.clone();
    newer[8] = 2;
    assert_eq!(Program::parse(&newer).unwrap_err(), Error::UnsupportedVersion(2));
    // Any changed byte fails the checksum.
    for position in (12..bytes.len() - 4).step_by(7) {
        let mut damaged = bytes.clone();
        damaged[position] ^= 0x40;
        assert_eq!(Program::parse(&damaged).unwrap_err(), Error::Checksum, "byte {position}");
    }
    // Changes with a fixed-up checksum reach the structural checks: they parse or are refused, and what parses
    // runs without a panic.
    let mut rng = Rng(0x5eed);
    let mut refused = 0;
    for _ in 0..20_000 {
        let mut damaged = bytes.clone();
        let body = damaged.len() - 4;
        for _ in 0..1 + rng.next() % 3 {
            let position = 12 + (rng.next() as usize) % (body - 12);
            damaged[position] = match rng.next() % 4 {
                0 => 0,
                1 => 0xFF,
                2 => damaged[position].wrapping_add(1),
                _ => rng.next() as u8,
            };
        }
        let sum = handwired_rs_run::checksum(&damaged[..body]);
        damaged[body..].copy_from_slice(&sum.to_le_bytes());
        match Program::parse(&damaged) {
            Err(_) => refused += 1,
            Ok(program) => {
                let inputs: Vec<f32> = (0..program.input_count())
                    .map(|cell| program.input_range(cell).map_or(0.0, |r| r.0 as f32))
                    .collect();
                if program.scratch_len() < 1 << 24 {
                    let _ = evaluate(&program, &inputs);
                }
            }
        }
    }
    assert!(refused > 1_000, "only {refused} damaged programs refused");
}
