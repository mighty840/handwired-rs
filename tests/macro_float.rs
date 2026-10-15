//! `#[handwired]` on f32 code: results must round exactly where Rust rounds.

use handwired_rs::handwired;

/// Float addition isn't associative, so each `+` must round in written order.
#[handwired]
fn ordered_sums(#[range(-1e6, 1e6)] a: f32, #[range(-1.0, 1.0)] b: f32, #[range(-1e-3, 1e-3)] c: f32) -> (f32, f32, f32) {
    let left = a + b + c;
    let right = a + (b + c);
    let mixed = a * 0.1 - b * c + 3.25;
    (left, right, mixed)
}

#[test]
fn sums_round_in_written_order() {
    let compiled = ordered_sums_circuit();
    let report = compiled.verify(50_000).unwrap();
    assert_eq!(report.skipped, 0);
    // A case where the two orders round differently in f32.
    // At 1e6 an f32 step is 0.0625: b alone is half a step and rounds away, b + c is more than half and doesn't.
    let args = (1e6f32, 0.03125, 0.001);
    let (left, right, _) = ordered_sums(1e6, 0.03125, 0.001);
    assert_ne!(left, right, "the test inputs should make the orders differ");
    assert_eq!(compiled.eval(&args).unwrap(), compiled.oracle(args));
}

#[handwired]
fn shapes(#[range(-100, 100)] x: f32, #[range(-100, 100)] y: f32) -> (f32, f32, f32, f32, f32, bool) {
    let product = x * y;
    let half = x / 2.0;
    let bounded = x.clamp(-10.0, 10.0);
    let smaller = x.min(y);
    let distance = (x - y).abs();
    (product, half, bounded, smaller, distance, x * x + y * y < 2500.0)
}

#[test]
fn products_clamps_and_comparisons() {
    assert_eq!(shapes_circuit().verify(50_000).unwrap().skipped, 0);
}

/// A u32 is wider than f32 holds exactly, so it needs a declared range.
#[handwired]
fn wide(#[range(0, 1_000_000)] a: u32, b: u16) -> u32 {
    a + b as u32 * 2
}

#[test]
fn wide_integers_with_ranges() {
    assert_eq!(wide_circuit().verify(10_000).unwrap().skipped, 0);
}

#[handwired]
fn too_wide(#[range(0, 16_000_000)] a: u32) -> u32 {
    a * 4
}

#[test]
fn values_past_two_to_the_24_fail_the_build() {
    let error = std::panic::catch_unwind(too_wide_circuit).err().unwrap();
    let message = error.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(message.contains("beyond ±2^24"), "{message}");
    // The message points at the user's code.
    assert!(message.contains("macro_float.rs"), "{message}");
}
