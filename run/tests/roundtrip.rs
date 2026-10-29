//! The format's contract: a program exported by handwired-rs and run here computes every value bit for bit as
//! handwired-rs's `TensorExecutor` does, and damaged bytes are refused, never a panic.

use handwired_rs::dsl::HwType;
use handwired_rs::{
    emit_argmax, emit_first_active, handwired, Circuit, OutOfRange, TensorExecutor, TensorProgram, Wire,
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

/// Input records the runner accepts: each range's ends, then random values on the cell's grid. A variant's cells
/// hold 0 where its tag doesn't select it, and an enum with gaps between its discriminants takes one of them.
fn records(program: &TensorProgram, count: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng(seed | 1);
    (0..count + 2)
        .map(|round| {
            let mut record: Vec<f32> = Vec::new();
            let mut selected = Vec::new();
            for rule in &program.source_rules[..program.source_size] {
                let cell = record.len();
                let (on, range) = match rule.variant {
                    Some((tag, variant, range)) => (selected[tag] && record[tag] == variant as f32, range),
                    None => (true, program.ranges[cell]),
                };
                let value = match (&rule.values, round) {
                    _ if !on => 0.0,
                    (Some(values), 0) => values[0],
                    (Some(values), 1) => values[values.len() - 1],
                    (Some(values), _) => values[(rng.unit() * values.len() as f64) as usize],
                    (None, 0) => range.min as f32,
                    (None, 1) => range.max as f32,
                    (None, _) => {
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
                    }
                };
                selected.push(on);
                record.push(value);
            }
            record
        })
        .collect()
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
fn inputs_off_their_grids_are_refused() {
    let program = plan_circuit().program;
    let bytes = program.to_bytes("plan");
    let parsed = Program::parse(&bytes).unwrap();
    let tags: Vec<usize> =
        (0..program.source_size).filter_map(|cell| parsed.input_variant(cell)).map(|v| v.0).collect();
    // Every selected integer cell, at its range's ends: half a step off is refused, a whole step is not.
    for on_grid in records(&program, 0, 1) {
        evaluate(&parsed, &on_grid).unwrap();
        let selected = selected(&parsed, &on_grid);
        for cell in 0..program.source_size {
            assert_eq!(parsed.input_grid(cell), program.grids[cell]);
            let (low, high) = parsed.input_range(cell).unwrap();
            if program.grids[cell] != Some(1.0) || low == high || !selected[cell] {
                continue;
            }
            let toward = if on_grid[cell] as f64 == high { -0.5 } else { 0.5 };
            let mut record = on_grid.clone();
            record[cell] += toward;
            let error = evaluate(&parsed, &record).unwrap_err();
            assert_eq!(error, Error::InputOffGrid { index: cell, value: record[cell], step: 1.0 });
            // A tag moved a whole step selects another variant, whose cells may not be 0.
            if !tags.contains(&cell) {
                record[cell] += toward;
                evaluate(&parsed, &record).unwrap();
            }
        }
    }
    // Floats have no grid.
    let physics = physics_circuit().program.to_bytes("physics");
    let physics = Program::parse(&physics).unwrap();
    assert_eq!(physics.input_grid(0), None);
    evaluate(&physics, &[0.3, -1.7, 0.123]).unwrap();
}

/// Whether each cell of `record` is selected: not a variant's field, or its tag holds the variant and is selected.
fn selected(parsed: &Program, record: &[f32]) -> Vec<bool> {
    let mut selected = Vec::new();
    for cell in 0..record.len() {
        selected.push(match parsed.input_variant(cell) {
            Some((tag, variant)) => selected[tag] && record[tag] == variant as f32,
            None => true,
        });
    }
    selected
}

#[test]
fn cells_of_unselected_variants_are_refused() {
    // plan's first action is Wait, Move { dx } or Attack(damage), in cells 0, 1 and 2.
    let program = plan_circuit().program;
    let bytes = program.to_bytes("plan");
    let parsed = Program::parse(&bytes).unwrap();
    assert_eq!(
        (parsed.input_variant(0), parsed.input_variant(1), parsed.input_variant(2)),
        (None, Some((0, 1)), Some((0, 2)))
    );
    let mut record = records(&program, 0, 1).remove(0);
    assert_eq!(record[..3], [0.0, 0.0, 0.0]);
    record[2] = 3.0;
    let error = evaluate(&parsed, &record).unwrap_err();
    assert_eq!(error, Error::InputOutsideVariant { index: 2, value: 3.0, tag: 0, variant: 2 });
    record[0] = 2.0;
    evaluate(&parsed, &record).unwrap();
    // Where it's selected, a variant's cell keeps its declared range, without the 0 it holds elsewhere.
    assert_eq!(parsed.input_range(2), Some((1.0, 6.0)));
    record[2] = 0.0;
    assert_eq!(evaluate(&parsed, &record).unwrap_err(), Error::InputOutOfRange { index: 2, value: 0.0 });
    // In random accepted records, any unselected cell set is refused.
    for record in records(&program, 200, 5) {
        for (cell, selected) in selected(&parsed, &record).into_iter().enumerate() {
            if !selected {
                let mut set = record.clone();
                set[cell] = 1.0;
                assert!(
                    matches!(evaluate(&parsed, &set), Err(Error::InputOutsideVariant { index, .. }) if index == cell)
                );
            }
        }
    }
}

/// Variants within variants.
#[handwired]
fn nested(value: Option<Result<u8, u8>>, #[range(0, 3)] shift: u8) -> u8 {
    match value {
        Some(Ok(x)) => x >> shift,
        Some(Err(x)) => x.min(9),
        None => 0,
    }
}

#[test]
fn nested_variants_are_selected_only_where_every_tag_selects_them() {
    let program = nested_circuit().program;
    roundtrip("nested", &program, 200);
    let bytes = program.to_bytes("nested");
    let parsed = Program::parse(&bytes).unwrap();
    let variants: Vec<_> = (0..5).map(|cell| parsed.input_variant(cell)).collect();
    assert_eq!(variants, [None, Some((0, 1)), Some((1, 0)), Some((1, 1)), None]);
    assert_eq!(evaluate(&parsed, &[1.0, 0.0, 200.0, 0.0, 2.0]).unwrap(), [50.0]);
    // None, whose Result tag is 0, Ok's tag: the Ok cell is still not selected.
    let error = evaluate(&parsed, &[0.0, 0.0, 200.0, 0.0, 2.0]).unwrap_err();
    assert_eq!(error, Error::InputOutsideVariant { index: 2, value: 200.0, tag: 0, variant: 1 });
    let error = evaluate(&parsed, &[1.0, 1.0, 200.0, 0.0, 2.0]).unwrap_err();
    assert_eq!(error, Error::InputOutsideVariant { index: 2, value: 200.0, tag: 1, variant: 0 });
    let error = evaluate(&parsed, &[0.0, 1.0, 0.0, 0.0, 2.0]).unwrap_err();
    assert_eq!(error, Error::InputOutsideVariant { index: 1, value: 1.0, tag: 0, variant: 1 });
}

/// A fieldless enum with gaps between its discriminants.
#[derive(Clone, Copy, Debug, PartialEq, HwType)]
enum Tier {
    Low = 2,
    Mid = 5,
    High = 40,
}

#[handwired]
fn tiers(tier: Tier, spare: Option<Tier>) -> (u8, bool) {
    let high = match spare {
        Some(other) => other == Tier::High,
        None => false,
    };
    (tier as u8 + 1, high)
}

#[test]
fn values_between_an_enums_discriminants_are_refused() {
    let program = tiers_circuit().program;
    roundtrip("tiers", &program, 200);
    let bytes = program.to_bytes("tiers");
    let parsed = Program::parse(&bytes).unwrap();
    let values = |cell| parsed.input_values(cell).map(|values| values.collect::<Vec<_>>());
    assert_eq!((values(0), values(1), values(2)), (Some(vec![2.0, 5.0, 40.0]), None, Some(vec![2.0, 5.0, 40.0])));
    assert_eq!((parsed.input_variant(2), parsed.input_range(2)), (Some((1, 1)), Some((2.0, 40.0))));
    assert_eq!(evaluate(&parsed, &[5.0, 1.0, 40.0]).unwrap(), [6.0, 1.0]);
    assert_eq!(evaluate(&parsed, &[6.0, 0.0, 0.0]).unwrap_err(), Error::InputNotAllowed { index: 0, value: 6.0 });
    assert_eq!(evaluate(&parsed, &[5.0, 1.0, 39.0]).unwrap_err(), Error::InputNotAllowed { index: 2, value: 39.0 });
    assert_eq!(evaluate(&parsed, &[5.0, 1.0, 0.0]).unwrap_err(), Error::InputOutOfRange { index: 2, value: 0.0 });
}

/// `program` as version 1 or 2 bytes: each source's range (with the 0 a variant's cells hold elsewhere) and, in
/// version 2, its step; no variants or value sets.
fn older(program: &TensorProgram, name: &str, bytes: &[u8], version: u32) -> Vec<u8> {
    let header = 8 + 4 + 4 + name.len();
    let mut old = bytes[..header + 5 * 4].to_vec();
    old[8..12].copy_from_slice(&version.to_le_bytes());
    for cell in 0..program.source_size {
        old.extend_from_slice(&program.ranges[cell].min.to_le_bytes());
        old.extend_from_slice(&program.ranges[cell].max.to_le_bytes());
        if version == 2 {
            old.extend_from_slice(&program.grids[cell].unwrap_or(0.0).to_le_bytes());
        }
    }
    let value_sets: usize =
        program.source_rules.iter().filter_map(|rule| rule.values.as_ref()).map(|values| 8 + 4 * values.len()).sum();
    old.extend_from_slice(&bytes[header + 6 * 4 + 32 * program.source_size + value_sets..bytes.len() - 4]);
    let sum = handwired_rs_run::checksum(&old);
    old.extend_from_slice(&sum.to_le_bytes());
    old
}

/// Versions 1 and 2 still run, with the checks they carry: version 1 none of grids, neither any of variants.
#[test]
fn programs_of_older_versions_run_with_the_checks_they_carry() {
    for (name, program) in [("plan", plan_circuit().program), ("tiers", tiers_circuit().program)] {
        let bytes = program.to_bytes(name);
        let current = Program::parse(&bytes).unwrap();
        for version in [1, 2] {
            let old = older(&program, name, &bytes, version);
            let parsed = Program::parse(&old).unwrap_or_else(|error| panic!("{name} version {version}: {error}"));
            for record in records(&program, 50, 3) {
                assert_eq!(evaluate(&parsed, &record), evaluate(&current, &record));
            }
            assert_eq!((parsed.input_variant(2), parsed.input_values(0).is_some()), (None, false));
            assert_eq!(parsed.input_grid(0), if version == 1 { None } else { Some(1.0) });
        }
    }
    let program = plan_circuit().program;
    let bytes = program.to_bytes("plan");
    let mut record = records(&program, 0, 1).remove(0);
    record[2] = 3.0;
    let mut half = records(&program, 0, 1).remove(0);
    half[12] += 0.5;
    for version in [1, 2] {
        let parsed_bytes = older(&program, "plan", &bytes, version);
        let parsed = Program::parse(&parsed_bytes).unwrap();
        assert!(evaluate(&parsed, &record).is_ok());
        assert_eq!(evaluate(&parsed, &half).is_ok(), version == 1);
    }
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
    newer[8] = 4;
    assert_eq!(Program::parse(&newer).unwrap_err(), Error::UnsupportedVersion(4));
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

/// A version 1 program by hand: one input in 0..=0, one layer with a one-hot group over it and one read of `count`
/// candidates, all the input, gather (`index` false) or index; its output is the read's row.
fn one_read(index: bool, count: u32) -> Vec<u8> {
    let mut words: Vec<u32> = vec![1, 0, 1, 2, 1, 1, 1]; // version, description, sources, values, slots, roots, layers
    words.extend([0, 0, 0, 0]); // the input's range, as two f64 zeros
    words.push(1); // the output: the layer's row
    words.extend([1, 1, 0, 0, 0, 1, !index as u32, index as u32]); // output size, rows, entries, ..., reads
    words.extend([0, 0, 0]); // the bias and row offsets
    words.extend([0, 1, 0]); // the one-hot group: slot 0, over the input
    words.extend([0, 0, 0, count]); // row 0, slot or index row 0, mask or flags 0
    words.resize(words.len() + count as usize, 0);
    let mut bytes = b"HWCIRC\0\0".to_vec();
    bytes.extend(words.iter().flat_map(|word| word.to_le_bytes()));
    let sum = handwired_rs_run::checksum(&bytes);
    bytes.extend(sum.to_le_bytes());
    bytes
}

#[test]
fn reads_have_at_most_2_to_the_24_candidates() {
    for index in [false, true] {
        for count in [1, 1 << 24] {
            let bytes = one_read(index, count);
            let program = Program::parse(&bytes).unwrap_or_else(|error| panic!("{count} candidates: {error}"));
            assert_eq!(evaluate(&program, &[0.0]).unwrap(), [0.0]);
        }
        assert!(matches!(Program::parse(&one_read(index, (1 << 24) + 1)), Err(Error::Malformed(_))));
    }
}
