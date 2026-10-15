//! Smoke tests for the smaller builder functions.

use handwired_rs::{Circuit, emit_clamp_exact, emit_maximum, emit_or, emit_round};

#[test]
fn round_matches_rust_round_ties_even() {
    let mut circuit = Circuit::new(1);
    let x = circuit.source(0);
    circuit.assume_range(x, -1_000.0, 1_000.0);
    let rounded = emit_round(&mut circuit, x);
    for value in [-2.5f32, -1.5, -0.5, 0.5, 1.5, 2.5, 0.49999997, 3.7, -3.7, 999.5, 123.25] {
        let values = circuit.evaluate_reference(&[value]).unwrap();
        assert_eq!(values[rounded.index()], value.round_ties_even(), "round({value})");
    }
}

#[test]
fn or_over_every_bit_combination() {
    let mut circuit = Circuit::new(3);
    let bits: Vec<_> = (0..3).map(|index| circuit.source(index)).collect();
    for &bit in &bits {
        circuit.assume_bit(bit);
    }
    let any = emit_or(&mut circuit, &bits);
    for mask in 0..8u32 {
        let source: Vec<f32> = (0..3).map(|index| ((mask >> index) & 1) as f32).collect();
        let values = circuit.evaluate_reference(&source).unwrap();
        assert_eq!(values[any.index()], if mask != 0 { 1.0 } else { 0.0 });
    }
}

#[test]
fn clamp_exact_returns_an_operand_and_maximum_wins_when_bounds_cross() {
    let mut circuit = Circuit::new(3);
    let (value, low, high) = (circuit.source(0), circuit.source(1), circuit.source(2));
    for wire in [value, low, high] {
        circuit.assume_range(wire, -100.0, 100.0);
    }
    let clamped = emit_clamp_exact(&mut circuit, value, low, high);
    let larger = emit_maximum(&mut circuit, value, low);
    for (source, expected) in
        [([5.3, 0.1, 9.9], 5.3f32), ([-7.7, 0.1, 9.9], 0.1), ([50.5, 0.1, 9.9], 9.9), ([5.0, 9.0, 2.0], 2.0)]
    {
        let values = circuit.evaluate_reference(&source).unwrap();
        assert_eq!(values[clamped.index()], expected, "clamp {source:?}");
        assert_eq!(values[larger.index()], source[0].max(source[1]), "max {source:?}");
    }
}
