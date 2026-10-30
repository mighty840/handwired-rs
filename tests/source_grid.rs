//! A source declared on a grid must stay where FP32 holds every value of that grid: within 2^24 steps of 0.

use handwired_rs::Circuit;

const LIMIT: f64 = 16_777_216.0;

fn declare(min: f64, max: f64, grid: f64, grid_first: bool) {
    let mut circuit = Circuit::new(1);
    let a = circuit.source(0);
    if grid_first {
        circuit.assume_grid(a, grid);
        circuit.assume_range(a, min, max);
    } else {
        circuit.assume_range(a, min, max);
        circuit.assume_grid(a, grid);
    }
}

#[test]
fn ranges_up_to_2_pow_24_steps_are_accepted() {
    for grid_first in [false, true] {
        declare(0.0, LIMIT, 1.0, grid_first);
        declare(-LIMIT, 0.0, 1.0, grid_first);
        declare(0.0, LIMIT + 0.5, 1.0, grid_first);
        declare(0.0, 2.0 * LIMIT, 2.0, grid_first);
        declare(-LIMIT / 4.0, LIMIT / 4.0, 0.25, grid_first);
    }
}

#[test]
#[should_panic(expected = "FP32 can't hold every value of that grid past 2^24 steps (16777216), e.g. 16777217")]
fn a_range_past_2_pow_24_is_refused() {
    declare(0.0, LIMIT + 2.0, 1.0, false);
}

#[test]
#[should_panic(expected = "e.g. 16777217")]
fn a_grid_declared_after_the_range_is_checked_too() {
    declare(0.0, LIMIT + 1.0, 1.0, true);
}

#[test]
#[should_panic(expected = "e.g. -16777217")]
fn the_negative_side_is_checked() {
    declare(-LIMIT - 1.0, 0.0, 1.0, false);
}

#[test]
#[should_panic(expected = "e.g. 4194304.25")]
fn the_limit_counts_grid_steps() {
    declare(0.0, LIMIT / 4.0 + 0.25, 0.25, false);
}

#[test]
fn ranges_without_a_grid_and_computed_values_are_not_checked() {
    // Without a grid a source promises only its range; a computed value is whatever f32 makes of it.
    let mut circuit = Circuit::new(1);
    let a = circuit.source(0);
    circuit.assume_range(a, 0.0, 4.0 * LIMIT);
    let b = circuit.affine(a * 2.0 + 1.0);
    assert_ne!(b, a);
    circuit.assume_grid(b, 1.0);
}
