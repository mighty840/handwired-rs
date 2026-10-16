//! A small deterministic random source, so the tests need no dependencies.
#![allow(dead_code)]

use handwired_rs::{Circuit, CompareOptions, Lin, OutOfRange, Relation, Wire, emit_compare_with};

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in 0..bound.
    pub fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    /// Uniform in min..=max.
    pub fn uniform(&mut self, min: f64, max: f64) -> f64 {
        min + (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * (max - min)
    }

    pub fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())]
    }
}

/// The next f32 toward +infinity or -infinity.
pub fn step_f32(value: f32, up: bool) -> f32 {
    if value == 0.0 {
        let tiny = f32::from_bits(1);
        return if up { tiny } else { -tiny };
    }
    let bits = value.to_bits();
    let away_from_zero = (value > 0.0) == up;
    f32::from_bits(if away_from_zero { bits + 1 } else { bits - 1 })
}

pub const SOURCES: usize = 6;

/// A random circuit over 4 values within ±10, one bit, and one integer index within 0..7.
pub fn random_circuit(rng: &mut Rng, operations: usize) -> Circuit {
    let mut circuit = Circuit::new(SOURCES);
    for index in 0..4 {
        circuit.assume_range(circuit.source(index), -10.0, 10.0);
    }
    circuit.assume_bit(circuit.source(4));
    let index = circuit.source(5);
    circuit.assume_range(index, 0.0, 7.0);
    circuit.assume_grid(index, 1.0);
    let bit = circuit.source(4);
    let coefficients = [1.0, -1.0, 0.5, 2.0, -0.25, 0.1, 3.7, -1.3];
    let mut wires: Vec<Wire> = (0..4).map(|index| circuit.source(index)).collect();
    while circuit.operations().len() < operations {
        let bounded: Vec<Wire> = wires
            .iter()
            .copied()
            .filter(|&wire| {
                let range = circuit.range(wire);
                range.is_finite() && range.min.abs().max(range.max.abs()) < 1e4
            })
            .collect();
        let pick = |rng: &mut Rng| rng.pick(&bounded);
        let form = |rng: &mut Rng| {
            let count = 1 + rng.below(5);
            let terms = (0..count).map(|_| (pick(rng), rng.pick(&coefficients))).collect();
            Lin::new(terms, rng.pick(&coefficients))
        };
        let wire = match rng.below(7) {
            0 => circuit.affine(form(rng)),
            1 => circuit.round_affine(form(rng)),
            2 => circuit.reglu(form(rng), form(rng)),
            3 => circuit.multiply(pick(rng), pick(rng)),
            4 => circuit.select(bit, pick(rng), pick(rng)),
            5 => {
                let (a, b) = (pick(rng), pick(rng));
                let relation = rng.pick(&[Relation::Greater, Relation::AtLeast, Relation::Equal]);
                let end_with_boolean = rng.below(2) == 0;
                emit_compare_with(&mut circuit, a, relation, b, CompareOptions { end_with_boolean })
            }
            _ => {
                let table: Vec<Wire> = (0..1 + rng.below(8)).map(|_| pick(rng)).collect();
                let rule = rng.pick(&[OutOfRange::Clamp, OutOfRange::Zero]);
                circuit.index_read(index, &table, rule)
            }
        };
        wires.push(wire);
    }
    circuit
}

pub fn random_source(rng: &mut Rng) -> Vec<f32> {
    let mut source: Vec<f32> = (0..4).map(|_| rng.uniform(-10.0, 10.0) as f32).collect();
    source.push(rng.below(2) as f32);
    source.push(rng.below(8) as f32);
    source
}
