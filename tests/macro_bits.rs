//! Integer bitwise operators and shifts in `#[handwired]` functions. The 8-bit
//! cases run every input through the circuit; the rest are verified.

use handwired_rs::handwired;

/// Every operator on two u8s.
#[handwired]
fn u8_ops(a: u8, b: u8) -> (u8, u8, u8, u8, u8, u8, u32) {
    (a & b, a | b, a ^ b, !a, a << 3, a >> 2, a.count_ones())
}

/// Every operator on two i8s, where `>>` is arithmetic and `<<` wraps into the sign.
#[handwired]
fn i8_ops(a: i8, b: i8) -> (i8, i8, i8, i8, i8, i8, u32) {
    (a & b, a | b, a ^ b, !a, a << 2, a >> 3, a.count_ones())
}

#[test]
fn eight_bit_operators_match_rust_on_every_input() {
    let pairs: Vec<(u8, u8)> = (0..=255u8).flat_map(|a| (0..=255u8).map(move |b| (a, b))).collect();
    let compiled = u8_ops_circuit();
    let results = compiled.run_batch(&pairs).unwrap();
    for (&(a, b), result) in pairs.iter().zip(&results) {
        assert_eq!(*result, u8_ops(a, b), "{a} {b}");
    }
    let pairs: Vec<(i8, i8)> = (-128..=127i8).flat_map(|a| (-128..=127i8).map(move |b| (a, b))).collect();
    let compiled = i8_ops_circuit();
    let results = compiled.run_batch(&pairs).unwrap();
    for (&(a, b), result) in pairs.iter().zip(&results) {
        assert_eq!(*result, i8_ops(a, b), "{a} {b}");
    }
}

/// One u8 XOR: each side split into 8 bits, an AND per bit position, and one sum.
#[handwired]
fn xor(a: u8, b: u8) -> u8 {
    a ^ b
}

#[test]
fn an_eight_bit_xor_costs_one_compare_per_bit_and_one_and_per_bit() {
    let compiled = xor_circuit();
    assert_eq!(compiled.verify(2_000).unwrap().skipped, 0);
    let (operations, layers) = (compiled.program.live_operation_count(), compiled.program.layers.len());
    // Each side's split is 7 steps of 2 ReLUs (the last bit is what's left), one layer each; then 8 ANDs and the sum.
    assert!(operations <= 40, "{operations} operations");
    assert!(layers <= 10, "{layers} layers");
}

/// Values of 11 bits split into bits that readers copy; 12 bits into bits kept as their own sums.
#[handwired]
fn eleven_bits(#[range(-2048, 2047)] a: i16, #[range(-2048, 2047)] b: i16) -> (i16, i16, i16, i16, i16, u32) {
    (a ^ b, a & b, a | b, a >> 5, a << 5, a.count_ones())
}

#[handwired]
fn twelve_bits(#[range(0, 4095)] a: u16, #[range(0, 4095)] b: u16) -> (u16, u16, u16, u16, u16, u32) {
    (a ^ b, a & b, a | b, a >> 5, a << 5, a.count_ones())
}

#[test]
fn both_split_modes_match_rust_on_every_value_of_one_side() {
    let others = [-2048i16, -2047, -1365, -1, 0, 1, 7, 1023, 1024, 2047];
    let records: Vec<(i16, i16)> = (-2048..=2047i16).flat_map(|a| others.map(|b| (a, b))).collect();
    let results = eleven_bits_circuit().run_batch(&records).unwrap();
    for (&(a, b), result) in records.iter().zip(&results) {
        assert_eq!(*result, eleven_bits(a, b), "{a} {b}");
    }
    let others = [0u16, 1, 2730, 2047, 2048, 4094, 4095];
    let records: Vec<(u16, u16)> = (0..=4095u16).flat_map(|a| others.map(|b| (a, b))).collect();
    let results = twelve_bits_circuit().run_batch(&records).unwrap();
    for (&(a, b), result) in records.iter().zip(&results) {
        assert_eq!(*result, twelve_bits(a, b), "{a} {b}");
    }
}

/// Shifts by an amount that depends on the inputs, of both widths and signs.
#[handwired]
fn variable_shifts(
    x: u8,
    #[range(0, 7)] s: u8,
    #[range(-3000, 3000)] y: i16,
    #[range(0, 15)] t: u32,
) -> (u8, u8, i16, i16) {
    (x << s, x >> s, y << t, y >> t)
}

#[test]
fn shifts_by_circuit_amounts() {
    let compiled = variable_shifts_circuit();
    let report = compiled.verify(20_000).unwrap();
    assert!(report.checked > 10_000, "{report:?}");
    assert_eq!(compiled.eval(&(0b1011_0001, 3, -1234, 4)).unwrap(), (0b1000_1000, 0b0001_0110, -19744, -78));
}

/// Native operands on either side and every compound assignment.
#[handwired]
fn mixed(x: u8, #[range(-100, 100)] y: i16) -> (u8, i16) {
    let mut a = x & 0x0F;
    a |= 0xF0 & x;
    a ^= 0x5A;
    a <<= 1;
    a >>= 2;
    let mut b = y;
    b ^= -1;
    b &= 0x7F;
    b |= 1 << 9;
    b = (0x3 ^ b) >> 1;
    (a, b)
}

#[test]
fn native_operands_and_compound_assignments() {
    let compiled = mixed_circuit();
    for x in 0..=255u8 {
        for y in [-100i16, -37, -1, 0, 1, 64, 99, 100] {
            assert_eq!(compiled.eval(&(x, y)).unwrap(), mixed(x, y), "{x} {y}");
        }
    }
}

/// xorshift16 (Marsaglia's 7, 9, 8 triple): shifts that wrap the u16 every step.
#[handwired]
fn xorshift16(state: u16) -> u16 {
    let mut x = state;
    x ^= x << 7;
    x ^= x >> 9;
    x ^= x << 8;
    x
}

#[test]
fn xorshift_matches_on_every_state() {
    let compiled = xorshift16_circuit();
    let states: Vec<(u16,)> = (0..=u16::MAX).map(|state| (state,)).collect();
    let results = compiled.run_batch(&states).unwrap();
    for (&(state,), &result) in states.iter().zip(&results) {
        assert_eq!(result, xorshift16(state), "{state}");
    }
}

const WIN_LINES: [u16; 8] = [0o007, 0o070, 0o700, 0o111, 0o222, 0o444, 0o421, 0o124];

/// Tic-tac-toe on bitboards: place a mark at a computed square if it's free, then test the eight lines with masks.
#[handwired]
fn bitboard_move(#[range(0, 511)] mine: u16, #[range(0, 511)] theirs: u16, #[range(0, 8)] square: u8) -> (u16, bool) {
    let bit = 1u16 << square;
    let mut next = mine;
    if (mine | theirs) & bit == 0 {
        next |= bit;
    }
    let mut won = false;
    for line in WIN_LINES {
        if next & line == line {
            won = true;
        }
    }
    (next, won)
}

#[test]
fn bitboards_place_and_test_lines() {
    let compiled = bitboard_move_circuit();
    let report = compiled.verify(20_000).unwrap();
    assert_eq!(report.skipped, 0);
    assert_eq!(compiled.eval(&(0b000_000_011, 0b000_011_000, 2)).unwrap(), (0b000_000_111, true));
    assert_eq!(compiled.eval(&(0b000_000_011, 0b000_011_100, 2)).unwrap(), (0b000_000_011, false));
}

/// Wider integer types work while their values stay within ±2^24.
#[handwired]
fn wide(#[range(0, 1_000_000)] a: u32, #[range(-500_000, 500_000)] b: i32) -> (u32, i32, i32, u32) {
    (a ^ (a >> 3), b & -16, b | 0xFF, a & 0xF0F0)
}

#[test]
fn wide_types_within_two_to_the_24() {
    let compiled = wide_circuit();
    assert_eq!(compiled.verify(20_000).unwrap().skipped, 0);
}

/// A u32 shifted past 2^24 can't be held exactly.
#[handwired]
fn too_wide(#[range(0, 1_000_000)] a: u32) -> u32 {
    a << 8
}

#[test]
#[should_panic(expected = "beyond ±2^24")]
fn shifting_past_two_to_the_24_is_rejected_at_build() {
    too_wide_circuit();
}
