//! `/` and `%` by values that depend on the inputs. Every circuit is verified
//! against the function it came from; division by zero panics in Rust, so
//! `verify` skips those inputs.

use handwired_rs::handwired;

/// Small ranges: each result is one read of a table over every pair.
#[handwired]
fn share(#[range(0, 200)] pot: u16, #[range(1, 6)] players: u16) -> (u16, u16) {
    (pot / players, pot % players)
}

#[test]
fn small_ranges_read_a_table_of_pairs() {
    let compiled = share_circuit();
    assert_eq!(compiled.verify(5_000).unwrap().skipped, 0);
    assert_eq!(compiled.eval(&(200, 6)).unwrap(), (33, 2));
    // The pair's index, then one read for each.
    assert_eq!(compiled.program.layers.len(), 2);
}

/// Signed, with divisors of either sign and zero among them: long division over magnitudes.
#[handwired]
fn signed(#[range(-60_000, 60_000)] a: i32, #[range(-300, 300)] b: i32) -> (i32, i32) {
    (a / b, a % b)
}

#[test]
fn signed_values_divide_toward_zero() {
    let compiled = signed_circuit();
    let report = compiled.verify(20_000).unwrap();
    assert!(report.checked > 15_000 && report.skipped > 0, "{report:?}");
    for (a, b) in [(-60_000, 7), (60_000, -7), (-59_999, -300), (299, 300), (-1, 1), (0, -5), (12_345, -1)] {
        assert_eq!(compiled.eval(&(a, b)).unwrap(), (a / b, a % b), "{a} / {b}");
    }
}

/// Divisors up to 2^23, of a narrow and of a wide dividend.
#[handwired]
fn large(#[range(0, 16_000_000)] x: u32, #[range(1, 8_388_608)] n: u32) -> (u32, u32, u32) {
    let hashed = x.wrapping_mul(2_654_435_761);
    (x / n, x % n, hashed % n)
}

#[test]
fn divisors_up_to_two_to_the_23_divide() {
    let compiled = large_circuit();
    assert_eq!(compiled.verify(3_000).unwrap().skipped, 0);
    for (x, n) in [(16_000_000, 8_388_608), (8_388_607, 8_388_608), (15_999_999, 3), (0, 1), (16_000_000, 1)] {
        assert_eq!(compiled.eval(&(x, n)).unwrap(), large(x, n), "{x} / {n}");
    }
}

/// A wide dividend by a small-range divisor, the quotient returned as bytes.
#[handwired]
fn hashed(#[range(0, 16_000_000)] x: u32, #[range(1, 1000)] n: u32) -> ([u8; 4], u32) {
    let hashed = x.wrapping_mul(2_654_435_761);
    let quotient = hashed / n;
    ([quotient as u8, (quotient >> 8) as u8, (quotient >> 16) as u8, (quotient >> 24) as u8], hashed % n)
}

#[test]
fn wide_dividends_divide_by_values() {
    let compiled = hashed_circuit();
    assert_eq!(compiled.verify(3_000).unwrap().skipped, 0);
    for (x, n) in [(15_999_999, 1), (12_345_678, 997), (1, 1000), (0, 3)] {
        assert_eq!(compiled.eval(&(x, n)).unwrap(), hashed(x, n));
    }
}

/// Literal on the left, `%=` and `/=`, and a u8 divisor cast up.
#[handwired]
fn compound(#[range(1, 50)] x: u32, #[range(0, 255)] step: u8) -> (u32, u32) {
    let mut left = 1_000_000 / x;
    left %= step as u32 + 1;
    let mut right = 5_000_000u32;
    right /= x;
    (left, right)
}

#[test]
fn literals_and_compound_assignments_divide() {
    let compiled = compound_circuit();
    assert_eq!(compiled.verify(5_000).unwrap().skipped, 0);
    assert_eq!(compiled.eval(&(7, 10)).unwrap(), compound(7, 10));
}

/// i8 division overflows on -128 / -1, which panics in Rust; a pair table reads 0 there.
#[handwired]
fn tiny(a: i8, #[range(-2, 2)] b: i8) -> i8 {
    a / b
}

#[test]
fn panicking_pairs_are_skipped() {
    let report = tiny_circuit().verify(5_000).unwrap();
    assert!(report.checked > 1_000 && report.skipped > 0, "{report:?}");
}

/// A divisor beyond 2^23 fails the build.
#[handwired]
fn too_large(#[range(0, 1000)] x: u32, #[range(1, 10_000_000)] d: u32) -> u32 {
    x / d
}

#[test]
fn divisors_beyond_two_to_the_23_are_rejected() {
    let error = std::panic::catch_unwind(too_large_circuit).err().unwrap();
    let message = error.downcast_ref::<String>().unwrap();
    assert!(message.contains("divisors up to 2^23"), "{message}");
}

/// A signed dividend over the full i32 width, as bytes, by a divisor of either sign.
#[handwired]
fn signed_wide(a: [u8; 4], #[range(-1000, 1000)] b: i32) -> ([u8; 4], i32) {
    let x = (a[0] as i32) | (a[1] as i32) << 8 | (a[2] as i32) << 16 | (a[3] as i32) << 24;
    let quotient = x / b;
    ([quotient as u8, (quotient >> 8) as u8, (quotient >> 16) as u8, (quotient >> 24) as u8], x % b)
}

#[test]
fn signed_wide_dividends_divide_by_values() {
    let compiled = signed_wide_circuit();
    let report = compiled.verify(3_000).unwrap();
    assert!(report.checked > 2_500, "{report:?}");
    for (x, b) in [(i32::MIN, 1000), (i32::MIN, -7), (i32::MAX, -1000), (-1, 3), (123_456_789, -1)] {
        let a = x.to_le_bytes();
        assert_eq!(compiled.eval(&(a, b)).unwrap(), signed_wide(a, b), "{x} / {b}");
    }
}

/// The full i16 range, whose minimum's magnitude doesn't fit an i16, by long division.
#[handwired]
fn full_i16(a: i16, #[range(-1000, 1000)] b: i16) -> (i16, i16) {
    (a / b, a % b)
}

#[test]
fn the_minimum_of_a_narrow_signed_type_divides() {
    let compiled = full_i16_circuit();
    let report = compiled.verify(5_000).unwrap();
    assert!(report.checked > 4_000, "{report:?}");
    for (a, b) in [(i16::MIN, 7), (i16::MIN, 1), (i16::MIN, -1000), (i16::MAX, -7), (i16::MIN, 1000)] {
        assert_eq!(compiled.eval(&(a, b)).unwrap(), (a / b, a % b), "{a} / {b}");
    }
}
