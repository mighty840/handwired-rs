//! Writes a suite for checking `handwired-rs-run` on a device (see `hardware/suite`): programs, input records within
//! their declared ranges, and the outputs handwired-rs's `TensorExecutor` computes for them, as bits.
//!
//! `cargo run -p handwired-rs-run --example hardware_suite -- OUT [RECORDS] [--small]` writes OUT with RECORDS random
//! records per program (default 200) plus each range's ends; `--small` leaves out the 84 KB wide hash, for chips with
//! little flash. `--corrupt` flips one expected bit in the first case, to see the check fail.

use handwired_rs::dsl::HwType;
use handwired_rs::{Circuit, TensorExecutor, TensorProgram, Wire, emit_argmax, emit_first_active, handwired};

/// A heater's duty from a temperature in tenths of a degree and a setpoint in degrees: firmware-style logic.
#[handwired]
fn heater(#[range(-400, 1250)] tenths: i16, #[range(5, 30)] setpoint: i16, boost: bool) -> (u8, bool) {
    let error = setpoint * 10 - tenths;
    let duty = if error <= 0 {
        0
    } else if error > 100 {
        100
    } else {
        error as u8
    };
    (if boost { duty.saturating_add(20).min(100) } else { duty }, error < -50)
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

/// Pair tables (table reads) and long division.
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

/// Fractional floats: products, sums, comparisons and a clamp.
#[handwired]
fn physics(#[range(-10.0, 10.0)] x: f32, #[range(-4.0, 4.0)] v: f32, #[range(0.0, 1.0)] drag: f32) -> (f32, f32, bool) {
    let next_v = (v - v * drag * 0.5).clamp(-3.0, 3.0);
    let next_x = x + next_v * 0.25;
    (next_x, next_v, next_x > 0.0)
}

/// Values small enough to be subnormal: an FPU that flushes them to zero gives different bits here, and only here.
#[handwired]
fn subnormals(#[range(-1e-36, 1e-36)] a: f32, #[range(0.0, 1.0)] b: f32) -> (f32, bool) {
    let product = a * b * 0.125;
    (product, product > 0.0)
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
    circuit.compile(Some(&[a, b, total]))
}

/// xorshift64*.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Input records within the declared ranges: each range's ends, then random values on the cell's grid.
fn records(program: &TensorProgram, count: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng(seed | 1);
    let cells = 0..program.source_size;
    let mut records = vec![
        cells.clone().map(|cell| program.ranges[cell].min as f32).collect(),
        cells.clone().map(|cell| program.ranges[cell].max as f32).collect(),
    ];
    for _ in 0..count {
        records.push(
            cells
                .clone()
                .map(|cell| {
                    let range = program.ranges[cell];
                    let value = range.min + rng.unit() * (range.max - range.min);
                    let value = match program.grids[cell] {
                        Some(step) => ((value / step).floor() * step).clamp(range.min, range.max),
                        None => value,
                    };
                    let cell_value = value as f32;
                    let inside = (cell_value as f64) >= range.min && (cell_value as f64) <= range.max;
                    if inside { cell_value } else { range.min as f32 }
                })
                .collect(),
        );
    }
    records
}

fn padded(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
    out.resize(out.len().next_multiple_of(4), 0);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let small = args.iter().any(|arg| arg == "--small");
    let corrupt = args.iter().any(|arg| arg == "--corrupt");
    let positional: Vec<&String> = args.iter().filter(|arg| !arg.starts_with("--")).collect();
    let Some(path) = positional.first() else {
        eprintln!("usage: hardware_suite OUT [RECORDS] [--small] [--corrupt]");
        std::process::exit(2);
    };
    let count: usize = positional.get(1).map_or(200, |count| count.parse().expect("RECORDS is a number"));

    let mut cases: Vec<(&str, TensorProgram)> = vec![
        ("heater", heater_circuit().program),
        ("bits", bits_circuit().program),
        ("divide", divide_circuit().program),
        ("plan", plan_circuit().program),
        ("physics", physics_circuit().program),
        ("gathers", gathers()),
        ("subnormals", subnormals_circuit().program),
    ];
    if !small {
        cases.push(("wide_hash", wide_hash_circuit().program));
    }
    let mut out = Vec::new();
    out.extend_from_slice(b"HWSUITE\0");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&(cases.len() as u32).to_le_bytes());
    let (mut scratch, mut outputs) = (0, 0);
    for (index, (name, program)) in cases.iter().enumerate() {
        let bytes = program.to_bytes(name);
        padded(&mut out, name.as_bytes());
        padded(&mut out, &bytes);
        let records = records(program, count, 0x5157 + index as u64);
        out.extend_from_slice(&(records.len() as u32).to_le_bytes());
        let mut executor = TensorExecutor::new(program);
        let mut expected = Vec::new();
        for record in &records {
            record.iter().for_each(|cell| out.extend_from_slice(&cell.to_le_bytes()));
            expected.extend(executor.run(record).iter().map(|value| value.to_bits()));
        }
        if corrupt && index == 0 {
            expected[0] ^= 1;
        }
        expected.iter().for_each(|bits| out.extend_from_slice(&bits.to_le_bytes()));
        let parsed = handwired_rs_run::Program::parse(&bytes).unwrap();
        scratch = scratch.max(parsed.scratch_len());
        outputs = outputs.max(parsed.output_count());
        println!(
            "{name}: {} bytes, {} layers, {} values, {} records",
            bytes.len(),
            program.layers.len(),
            program.value_count,
            records.len()
        );
    }
    std::fs::write(path, &out).unwrap();
    println!("{} bytes in {path}; firmware needs {scratch} scratch cells and {outputs} output cells", out.len());
}
