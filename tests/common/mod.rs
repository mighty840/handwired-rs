//! A small deterministic random source, so the tests need no dependencies.
#![allow(dead_code)]

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
