//! FP32 helpers shared by the builder, the evaluators and the compiler.
//!
//! Builder-side arithmetic (ranges, coefficients, biases) runs in f64 and
//! rounds to f32 at the same points as handwired's `Math.fround`, so a range
//! inferred here matches the one handwired infers. Evaluation runs in native
//! f32, which rounds every `+` and `*` exactly like `Math.fround` of the f64
//! result, and Rust never fuses them into a multiply-add on its own.

/// `Math.fround`: round an f64 to the nearest f32, ties to even.
pub(crate) fn fround(value: f64) -> f64 {
    value as f32 as f64
}

/// The range of values a wire may hold. An open side is infinite.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Range {
    pub min: f64,
    pub max: f64,
}

impl Range {
    pub const UNKNOWN: Range = Range { min: f64::NEG_INFINITY, max: f64::INFINITY };

    pub fn new(min: f64, max: f64) -> Range {
        Range { min, max }
    }

    pub fn is_finite(&self) -> bool {
        self.min.is_finite() && self.max.is_finite()
    }
}

/// FP32 holds every integer up to 2^24 exactly, so sums on a grid stay exact
/// within this many grid steps.
pub(crate) const EXACT_GRID_STEPS: f64 = 16_777_216.0;
pub(crate) fn smallest_grid() -> f64 {
    pow2(-149)
}
pub(crate) fn largest_grid() -> f64 {
    pow2(127)
}

/// 2^exponent as an f64, exact for every exponent an f64 can hold, including
/// subnormal ones, where `powi` would round through an overflowing reciprocal.
pub(crate) fn pow2(exponent: i32) -> f64 {
    if exponent > 1023 {
        f64::INFINITY
    } else if exponent >= -1022 {
        f64::from_bits(((exponent + 1023) as u64) << 52)
    } else if exponent >= -1074 {
        f64::from_bits(1u64 << (exponent + 1074))
    } else {
        0.0
    }
}

/// The largest power of two dividing a finite nonzero value, e.g. 4 for 12 and
/// 0.25 for 0.75.
pub(crate) fn lowest_power_of_two(value: f64) -> f64 {
    debug_assert!(value.is_finite() && value != 0.0);
    let bits = value.to_bits();
    let biased_exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    if biased_exponent == 0 {
        pow2(-1074 + fraction.trailing_zeros() as i32)
    } else {
        let significand = fraction | (1u64 << 52);
        pow2(biased_exponent - 1075 + significand.trailing_zeros() as i32)
    }
}

pub(crate) fn is_power_of_two_grid(step: f64) -> bool {
    step.is_finite() && step >= smallest_grid() && step <= largest_grid() && lowest_power_of_two(step) == step
}

/// The smallest integer e with 2^e >= value, for a finite positive value.
pub(crate) fn power_of_two_exponent_at_least(value: f64) -> i32 {
    let mut exponent = value.log2().ceil() as i32;
    while pow2(exponent) < value {
        exponent += 1;
    }
    while pow2(exponent - 1) >= value {
        exponent -= 1;
    }
    exponent
}

/// JavaScript's `Math.max(0, x)` on an f32: NaN stays NaN, and -0 becomes +0.
pub(crate) fn relu32(value: f32) -> f32 {
    if value > 0.0 {
        value
    } else if value.is_nan() {
        f32::NAN
    } else {
        0.0
    }
}

/// The boolean atom: `(score + 8,388,609) - 8,388,609` in FP32 rounds a
/// score within 0..1 to 0 below 0.5 and to 1 from 0.5 up.
pub(crate) fn evaluate_boolean(score: f32) -> f32 {
    const BASE: f32 = 8_388_609.0;
    (score + BASE) - BASE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowest_power_of_two_matches_handwired() {
        assert_eq!(lowest_power_of_two(12.0), 4.0);
        assert_eq!(lowest_power_of_two(0.75), 0.25);
        assert_eq!(lowest_power_of_two(-3.0), 1.0);
        assert_eq!(lowest_power_of_two(pow2(-149)), pow2(-149));
        assert_eq!(lowest_power_of_two(f64::from_bits(6)), pow2(-1073));
    }

    #[test]
    fn boolean_atom_rounds_at_one_half() {
        assert_eq!(evaluate_boolean(0.5), 1.0);
        assert_eq!(evaluate_boolean(0.5 - 2f32.powi(-25)), 0.0);
        assert_eq!(evaluate_boolean(0.0), 0.0);
        assert_eq!(evaluate_boolean(1.0), 1.0);
    }

    #[test]
    fn exponent_at_least() {
        assert_eq!(power_of_two_exponent_at_least(16_384.0), 14);
        assert_eq!(power_of_two_exponent_at_least(16_385.0), 15);
        assert_eq!(power_of_two_exponent_at_least(pow2(-149)), -149);
    }
}
