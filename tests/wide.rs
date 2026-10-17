//! Integers beyond ±2^24, held as bytes inside the circuit. Most functions
//! take and return bytes, so `verify` samples the full width.

use handwired_rs::handwired;

fn word(bytes: [u8; 4]) -> u32 {
    u32::from_le_bytes(bytes)
}

/// Every u32 operation, from and to bytes.
#[handwired]
#[allow(clippy::identity_op)]
fn u32_ops(a: [u8; 4], b: [u8; 4], #[range(0, 31)] s: u32) -> [[u8; 4]; 14] {
    let x = (a[0] as u32) | (a[1] as u32) << 8 | (a[2] as u32) << 16 | (a[3] as u32) << 24;
    let y = (b[0] as u32) | (b[1] as u32) << 8 | (b[2] as u32) << 16 | (b[3] as u32) << 24;
    let results = [
        x.wrapping_add(y),
        x.wrapping_sub(y),
        x.wrapping_mul(y),
        x & y,
        x | y,
        x ^ y,
        !x,
        x << 5,
        x >> 7,
        x.rotate_left(13),
        x.rotate_right(3),
        x << s,
        x >> s,
        x.count_ones()
            + if x < y { 100u32 } else { 0 }
            + if x == y { 1000u32 } else { 0 }
            + if x >= y { 10000u32 } else { 0 },
    ];
    let mut out = [[0u8; 4]; 14];
    for i in 0..14 {
        out[i] = [results[i] as u8, (results[i] >> 8) as u8, (results[i] >> 16) as u8, (results[i] >> 24) as u8];
    }
    out
}

#[test]
fn u32_operations_match_rust() {
    let compiled = u32_ops_circuit();
    assert_eq!(compiled.verify(20_000).unwrap().skipped, 0);
    // Carries across every byte boundary, the extremes, and equal operands.
    let edges = [0u32, 1, 0xFF, 0x100, 0xFFFF, 0x1_0000, 0xFF_FFFF, 0x100_0000, 0x7FFF_FFFF, 0x8000_0000, u32::MAX];
    for &x in &edges {
        for &y in &edges {
            for s in [0, 1, 8, 31] {
                let args = (x.to_le_bytes(), y.to_le_bytes(), s);
                assert_eq!(compiled.eval(&args).unwrap(), u32_ops(args.0, args.1, s), "{x:#x} {y:#x} {s}");
            }
        }
    }
    let _ = word;
}

/// Signed 32-bit: arithmetic right shifts, signed order, sign extension to i64 and narrowing to i16.
#[handwired]
#[allow(clippy::needless_range_loop)]
fn i32_ops(a: [u8; 4], b: [u8; 4]) -> (bool, bool, [u8; 4], [u8; 8], i16, i8) {
    let x = ((a[0] as u32) | (a[1] as u32) << 8 | (a[2] as u32) << 16 | (a[3] as u32) << 24) as i32;
    let y = ((b[0] as u32) | (b[1] as u32) << 8 | (b[2] as u32) << 16 | (b[3] as u32) << 24) as i32;
    let shifted = (x >> 9) as u32;
    let long = x as i64 * 3;
    let long = long as u64;
    let mut long_bytes = [0u8; 8];
    for i in 0..8 {
        long_bytes[i] = (long >> (8 * i)) as u8;
    }
    (
        x < y,
        x.wrapping_sub(y) >= 0,
        [shifted as u8, (shifted >> 8) as u8, (shifted >> 16) as u8, (shifted >> 24) as u8],
        long_bytes,
        x as i16,
        y as i8,
    )
}

#[test]
fn i32_operations_match_rust() {
    let compiled = i32_ops_circuit();
    assert_eq!(compiled.verify(20_000).unwrap().skipped, 0);
    for x in [i32::MIN, -1, 0, 1, i32::MAX, -0x100_0000, 0xFF_FFFF] {
        for y in [i32::MIN, -1, 0, 1, i32::MAX] {
            let args = (x.to_le_bytes(), y.to_le_bytes());
            assert_eq!(compiled.eval(&args).unwrap(), i32_ops(args.0, args.1), "{x} {y}");
        }
    }
}

/// 64-bit: shifts by a circuit amount up to 63, rotates, comparisons and a multiply.
#[handwired]
#[allow(clippy::needless_range_loop)]
fn u64_ops(a: [u8; 8], b: [u8; 8], #[range(0, 63)] s: u8) -> ([u8; 8], [u8; 8], [u8; 8], bool, u32) {
    let mut x = 0u64;
    let mut y = 0u64;
    for i in 0..8 {
        x |= (a[i] as u64) << (8 * i);
        y |= (b[i] as u64) << (8 * i);
    }
    let shifted = (x << s) ^ (y >> s);
    let rotated = x.rotate_left(29).wrapping_add(y.rotate_right(41));
    let product = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut out = [[0u8; 8]; 3];
    for i in 0..8 {
        out[0][i] = (shifted >> (8 * i)) as u8;
        out[1][i] = (rotated >> (8 * i)) as u8;
        out[2][i] = (product >> (8 * i)) as u8;
    }
    (out[0], out[1], out[2], x > y, (x ^ y).count_ones())
}

#[test]
fn u64_operations_match_rust() {
    let compiled = u64_ops_circuit();
    assert_eq!(compiled.verify(5_000).unwrap().skipped, 0);
    for (x, y) in
        [(u64::MAX, 0), (0, u64::MAX), (1 << 63, (1 << 63) - 1), (0x0123_4567_89AB_CDEF, 0x0123_4567_89AB_CDEF)]
    {
        for s in [0, 1, 32, 63] {
            let args = (x.to_le_bytes(), y.to_le_bytes(), s);
            assert_eq!(compiled.eval(&args).unwrap(), u64_ops(args.0, args.1, s), "{x:#x} {y:#x} {s}");
        }
    }
}

/// FNV-1a over 8 bytes: XOR, then multiply by the 32-bit FNV prime, which carries through every byte.
#[handwired]
fn fnv1a(bytes: [u8; 8]) -> [u8; 4] {
    let mut hash = 0x811C_9DC5u32;
    for byte in bytes {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    [hash as u8, (hash >> 8) as u8, (hash >> 16) as u8, (hash >> 24) as u8]
}

#[test]
fn fnv1a_matches_rust() {
    let compiled = fnv1a_circuit();
    assert_eq!(compiled.verify(20_000).unwrap().skipped, 0);
    // The reference value for "handwire" from the FNV spec's algorithm.
    let expected =
        b"handwire".iter().fold(0x811C_9DC5u32, |hash, &byte| (hash ^ byte as u32).wrapping_mul(0x0100_0193));
    assert_eq!(compiled.eval(&(*b"handwire",)).unwrap(), expected.to_le_bytes());
}

const ROUND_CONSTANTS: [u32; 4] = [0x5A82_7999, 0x6ED9_EBA1, 0x8F1B_BCDC, 0xCA62_C1D6];

/// Wide constants in a table read at a circuit index, and wide values chosen by an input-dependent `if`.
#[handwired]
fn pick(#[range(0, 3)] round: u8, flip: bool, a: [u8; 4]) -> [u8; 4] {
    let x = (a[0] as u32) | (a[1] as u32) << 8 | (a[2] as u32) << 16 | (a[3] as u32) << 24;
    let k = ROUND_CONSTANTS[round as usize];
    let chosen = if flip { k ^ x } else { k.wrapping_add(x) };
    [chosen as u8, (chosen >> 8) as u8, (chosen >> 16) as u8, (chosen >> 24) as u8]
}

#[test]
fn wide_tables_and_selects() {
    let compiled = pick_circuit();
    assert_eq!(compiled.verify(20_000).unwrap().skipped, 0);
}

/// Connect Four on a 49-bit bitboard (7 columns of 7 bits, the top bit of each column empty): four in a row in any
/// direction.
#[handwired]
fn connect_four(cells: [bool; 42]) -> bool {
    let mut board = 0u64;
    for column in 0..7 {
        for row in 0..6 {
            if cells[column * 6 + row] {
                board |= 1u64 << (column * 7 + row);
            }
        }
    }
    let mut won = false;
    for step in [1, 7, 6, 8] {
        let pairs = board & (board >> step);
        if pairs & (pairs >> (2 * step)) != 0 {
            won = true;
        }
    }
    won
}

#[test]
fn connect_four_bitboards() {
    let compiled = connect_four_circuit();
    assert_eq!(compiled.verify(20_000).unwrap().skipped, 0);
    let mut cells = [false; 42];
    for column in 2..6 {
        cells[column * 6 + 3] = true;
    }
    assert!(compiled.eval(&(cells,)).unwrap());
    cells[4 * 6 + 3] = false;
    assert!(!compiled.eval(&(cells,)).unwrap());
}

/// A wide result can't be returned as one cell.
#[handwired]
fn unnarrowed(a: u8) -> u32 {
    (a as u32) << 24
}

#[test]
#[should_panic(expected = "result may leave ±2^24")]
fn wide_results_must_be_narrowed() {
    unnarrowed_circuit();
}

/// Narrow `+` keeps its guardrail: a loose range fails the build instead of turning into bytes.
#[handwired]
fn loose(#[range(0, 10_000_000)] a: u32) -> u32 {
    a + a
}

#[test]
#[should_panic(expected = "declare narrower")]
fn plain_arithmetic_keeps_its_range_check() {
    loose_circuit();
}

/// Rotates and `wrapping_mul` on the narrow types: u8 and i16 rotate through bits, a u16 product beyond 2^24 goes
/// through bytes and comes back as a u16.
#[handwired]
fn narrow_ops(a: u8, b: i16, c: u16, d: u16) -> (u8, u8, i16, u16, i8) {
    (a.rotate_left(3), a.rotate_right(7), b.rotate_left(5), c.wrapping_mul(d), (a as i8).wrapping_mul(-7))
}

#[test]
fn narrow_rotates_and_wrapping_products_match_rust() {
    let compiled = narrow_ops_circuit();
    assert_eq!(compiled.verify(20_000).unwrap().skipped, 0);
    let records: Vec<(u8, i16, u16, u16)> =
        (0..=255u8).map(|a| (a, (a as i16 - 128).wrapping_mul(257), a as u16 * 257, 65_535 - a as u16 * 3)).collect();
    let results = compiled.run_batch(&records).unwrap();
    for (&(a, b, c, d), &result) in records.iter().zip(&results) {
        assert_eq!(result, narrow_ops(a, b, c, d), "{a} {b} {c} {d}");
    }
}

/// Negative one-wire values widened to bytes: sign-extended through every byte.
#[handwired]
#[allow(clippy::needless_range_loop)]
fn signed_widening(#[range(-1000, 1000)] a: i32, b: i16) -> ([u8; 4], [u8; 8], [u8; 4]) {
    let unsigned = a as u32;
    let long = b as u64;
    let shifted = a.wrapping_add(0x7FFF_FFFF);
    let mut long_bytes = [0u8; 8];
    for i in 0..8 {
        long_bytes[i] = (long >> (8 * i)) as u8;
    }
    (
        [unsigned as u8, (unsigned >> 8) as u8, (unsigned >> 16) as u8, (unsigned >> 24) as u8],
        long_bytes,
        [shifted as u8, (shifted >> 8) as u8, (shifted >> 16) as u8, (shifted >> 24) as u8],
    )
}

#[test]
fn negative_values_sign_extend_into_bytes() {
    let compiled = signed_widening_circuit();
    assert_eq!(compiled.verify(20_000).unwrap().skipped, 0);
    let records: Vec<(i32, i16)> = (i16::MIN..=i16::MAX).map(|b| ((b as i32).clamp(-1000, 1000), b)).collect();
    let results = compiled.run_batch(&records).unwrap();
    for (&(a, b), result) in records.iter().zip(&results) {
        assert_eq!(*result, signed_widening(a, b), "{a} {b}");
    }
}

/// Saturating arithmetic on a wide value would wrap first, so it fails the build.
#[handwired]
fn saturating_wide(a: u8) -> u8 {
    let big = (a as u32) << 24;
    (big.saturating_add(0x0100_0000) >> 24) as u8
}

#[test]
#[should_panic(expected = "saturating_add on a value beyond ±2^24")]
fn saturating_on_wide_values_is_rejected() {
    saturating_wide_circuit();
}

/// Division and remainder by constants and `abs` on wide values: byte digits for divisors up to 65,536, narrower digits
/// above, powers of two as shifts and masks, signed values truncating toward zero; remainders and small quotients
/// come back as one wire.
#[handwired]
#[allow(clippy::type_complexity, clippy::needless_range_loop, clippy::cast_abs_to_unsigned)]
fn dividing(a: [u8; 4], b: [u8; 8]) -> ([u8; 4], u32, [u8; 4], i32, u32, u32, [u8; 8], u32, [u8; 4], u32, i32) {
    let x = (a[0] as u32) | (a[1] as u32) << 8 | (a[2] as u32) << 16 | (a[3] as u32) << 24;
    let y = x as i32;
    let mut long = 0u64;
    for i in 0..8 {
        long |= (b[i] as u64) << (8 * i);
    }
    let tenth = x / 10;
    let signed = (y / -7) as u32;
    let magnitude = (y | 1).abs() as u32;
    let quotient = long / 1_000_003;
    let mut quotient_bytes = [0u8; 8];
    for i in 0..8 {
        quotient_bytes[i] = (quotient >> (8 * i)) as u8;
    }
    (
        [tenth as u8, (tenth >> 8) as u8, (tenth >> 16) as u8, (tenth >> 24) as u8],
        x % 1000,
        [signed as u8, (signed >> 8) as u8, (signed >> 16) as u8, (signed >> 24) as u8],
        y % 7,
        x / 65_536,
        x % 4096,
        quotient_bytes,
        (long % 999_983) as u32,
        [magnitude as u8, (magnitude >> 8) as u8, (magnitude >> 16) as u8, (magnitude >> 24) as u8],
        x / 5_000_000,
        y >> 20,
    )
}

#[test]
fn division_remainder_and_abs_on_wide_values() {
    let compiled = dividing_circuit();
    println!(
        "dividing: {} operations in {} layers",
        compiled.program.live_operation_count(),
        compiled.program.layers.len()
    );
    assert_eq!(compiled.verify(20_000).unwrap().skipped, 0);
    let words = [
        0u32,
        1,
        6,
        7,
        9,
        10,
        999,
        1000,
        0xFFFF,
        0x1_0000,
        4_999_999,
        5_000_000,
        0x7FFF_FFFF,
        0x8000_0000,
        0x8000_0001,
        u32::MAX,
    ];
    let longs = [0u64, 1_000_002, 1_000_003, 999_983, u32::MAX as u64, 1 << 40, u64::MAX - 1, u64::MAX];
    for &x in &words {
        for &long in &longs {
            let args = (x.to_le_bytes(), long.to_le_bytes());
            assert_eq!(compiled.eval(&args).unwrap(), dividing(args.0, args.1), "{x:#x} {long:#x}");
        }
    }
}

/// A divisor beyond 2^23 is refused with a message.
#[handwired]
fn huge_divisor(a: [u8; 4]) -> u8 {
    let x = (a[0] as u32) | (a[1] as u32) << 24;
    (x / 10_000_000) as u8
}

#[test]
#[should_panic(expected = "divisors up to 2^23")]
fn divisors_beyond_two_to_the_23_are_rejected() {
    huge_divisor_circuit();
}

/// Masks and shifts that bring a wide value back within ±2^24 give it one wire again: it is returned, compared, cast
/// to f32 and added without a cast; an `if` of two such values keeps their bytes for the `^` after it; a negative
/// value whose high bytes are all set narrows too; and arithmetic that may leave ±2^24 again goes back to bytes.
#[handwired]
#[allow(clippy::type_complexity)]
fn narrowed(a: [u8; 4], #[range(0, 7)] s: u8) -> (u32, u32, i32, bool, f32, u32, u32, u32, u64) {
    let x = (a[0] as u32) | (a[1] as u32) << 8 | (a[2] as u32) << 16 | (a[3] as u32) << 24;
    let h = x.wrapping_mul(0x9E37_79B1);
    let top = h >> 24;
    let low = h & 0xFFFF;
    let negative = (h as i32) | !0xFFFFi32;
    let spread = low * 0x10001;
    let merged = if s > 3 { top } else { low };
    let long = (h as u64) << 8 >> 40;
    (
        top,
        low,
        negative,
        top > 200 && low < 1000,
        top as f32 * 0.5,
        spread >> 20,
        merged ^ (h >> 16),
        top + low * 3,
        long,
    )
}

#[test]
fn values_back_within_range_narrow_again() {
    let compiled = narrowed_circuit();
    assert_eq!(compiled.verify(20_000).unwrap().skipped, 0);
    let edges = [0u32, 1, 0xFF, 0xFFFF, 0x00FF_FFFF, 0x0100_0000, 0x7FFF_FFFF, 0x8000_0000, u32::MAX];
    for &x in &edges {
        for s in [0, 7] {
            let args = (x.to_le_bytes(), s);
            assert_eq!(compiled.eval(&args).unwrap(), narrowed(args.0, s), "{x:#x} {s}");
        }
    }
}

/// One-wire values whose ranges are too wide for a table divide by long division too.
#[handwired]
fn loose_division(#[range(0, 16_000_000)] a: u32, #[range(-9_000_000, 9_000_000)] b: i32) -> (u32, u32, i32, i32) {
    (a / 3, a % 3, b / -1000, b % 1000)
}

#[test]
fn loose_ranges_divide_by_long_division() {
    let compiled = loose_division_circuit();
    assert_eq!(compiled.verify(20_000).unwrap().skipped, 0);
    for (a, b) in [(0, 0), (16_000_000, -9_000_000), (15_999_999, 9_000_000), (2, -999), (3, 1000), (1, -1001)] {
        assert_eq!(compiled.eval(&(a, b)).unwrap(), loose_division(a, b), "{a} {b}");
    }
}

/// Divisors at the edges of long division: a u64 with digits narrower than a byte, the largest divisor (2^23 - 1, one
/// bit per digit), and powers of two up to 2^23, which take a shift and a mask instead.
#[handwired]
#[allow(clippy::needless_range_loop)]
fn division_edges(a: [u8; 4], b: [u8; 8]) -> ([u8; 8], u32, u32, u32, u32, u32) {
    let x = (a[0] as u32) | (a[1] as u32) << 8 | (a[2] as u32) << 16 | (a[3] as u32) << 24;
    let mut long = 0u64;
    for i in 0..8 {
        long |= (b[i] as u64) << (8 * i);
    }
    let quotient = long / 5_000_000;
    let mut quotient_bytes = [0u8; 8];
    for i in 0..8 {
        quotient_bytes[i] = (quotient >> (8 * i)) as u8;
    }
    (quotient_bytes, (long % 5_000_000) as u32, x / 8_388_607, x % 8_388_607, x / 8_388_608, x % 8_388_608)
}

#[test]
fn division_edges_match_rust() {
    let compiled = division_edges_circuit();
    assert_eq!(compiled.verify(20_000).unwrap().skipped, 0);
    let words = [0u32, 8_388_606, 8_388_607, 8_388_608, 16_777_214, 0x7FFF_FFFF, u32::MAX];
    let longs = [0u64, 4_999_999, 5_000_000, u32::MAX as u64, u64::MAX];
    for &x in &words {
        for &long in &longs {
            let args = (x.to_le_bytes(), long.to_le_bytes());
            assert_eq!(compiled.eval(&args).unwrap(), division_edges(args.0, args.1), "{x:#x} {long:#x}");
        }
    }
}

/// Unsigned division by a power of two moves bytes and masks them: no long division, the same circuit as the shift and
/// mask written out.
#[handwired]
fn by_powers_of_two(a: [u8; 4]) -> (u32, u32) {
    let x = (a[0] as u32) | (a[1] as u32) << 8 | (a[2] as u32) << 16 | (a[3] as u32) << 24;
    (x / 65_536, x % 4096)
}

#[handwired]
fn shifted_and_masked(a: [u8; 4]) -> (u32, u32) {
    let x = (a[0] as u32) | (a[1] as u32) << 8 | (a[2] as u32) << 16 | (a[3] as u32) << 24;
    (x >> 16, x & 0xFFF)
}

#[test]
fn powers_of_two_divide_by_shifting() {
    let compiled = by_powers_of_two_circuit();
    assert_eq!(compiled.verify(5_000).unwrap().skipped, 0);
    let written_out = shifted_and_masked_circuit().program.live_operation_count();
    assert_eq!(compiled.program.live_operation_count(), written_out);
}
