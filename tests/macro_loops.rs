//! `while`, `while let` and `loop` with `#[max_iterations(N)]`: the circuit
//! runs N passes, and verify fails where the function would run more.

use handwired_rs::dsl::{HwType, VerifyError};
use handwired_rs::handwired;

/// Euclid's algorithm by subtraction: up to 254 passes for u8s.
#[handwired]
fn gcd(#[range(1, 255)] a: u8, #[range(1, 255)] b: u8) -> u8 {
    let (mut a, mut b) = (a, b);
    #[max_iterations(254)]
    while a != b {
        if a > b {
            a -= b;
        } else {
            b -= a;
        }
    }
    a
}

#[test]
fn gcd_matches_on_every_pair() {
    let compiled = gcd_circuit();
    println!("gcd: {} operations in {} layers", compiled.program.live_operation_count(), compiled.program.layers.len());
    let pairs: Vec<(u8, u8)> = (1..=255u8).flat_map(|a| (1..=255u8).map(move |b| (a, b))).collect();
    let results = compiled.run_batch(&pairs).unwrap();
    for (&(a, b), &result) in pairs.iter().zip(&results) {
        assert_eq!(result, gcd(a, b), "{a} {b}");
    }
}

/// The digit sum and digit count of a u16.
#[handwired]
fn digits(n: u16) -> (u8, u8) {
    let (mut n, mut sum, mut count) = (n, 0u8, 0u8);
    #[max_iterations(5)]
    while n > 0 {
        sum += (n % 10) as u8;
        n /= 10;
        count += 1;
    }
    (sum, count)
}

#[test]
fn digit_loops_match_on_every_value() {
    let compiled = digits_circuit();
    let values: Vec<(u16,)> = (0..=u16::MAX).map(|n| (n,)).collect();
    let results = compiled.run_batch(&values).unwrap();
    for (&(n,), &result) in values.iter().zip(&results) {
        assert_eq!(result, digits(n), "{n}");
    }
}

/// `loop` with an input-dependent `break`: the first index holding the target, or 8.
#[handwired]
fn find(values: [u8; 8], target: u8) -> u8 {
    let mut index = 0u8;
    #[max_iterations(9)]
    loop {
        if index == 8 || values[index as usize] == target {
            break;
        }
        index += 1;
    }
    index
}

#[test]
fn loop_with_break() {
    let compiled = find_circuit();
    assert_eq!(compiled.verify(20_000).unwrap().skipped, 0);
    assert_eq!(compiled.eval(&([5, 1, 9, 9, 2, 0, 0, 3], 9)).unwrap(), 2);
    assert_eq!(compiled.eval(&([5, 1, 9, 9, 2, 0, 0, 3], 4)).unwrap(), 8);
}

/// A phase of a game turn.
#[derive(Clone, Copy, Debug, PartialEq, HwType)]
enum Phase {
    Rolling,
    Done,
}

/// `while let` on an enum, with rolls from a fixed list: keep rolling until a 1 or a total of 20.
#[handwired]
fn press_luck(#[range(1, 6)] rolls: [u8; 8]) -> (u8, u8) {
    let (mut phase, mut total, mut turn) = (Phase::Rolling, 0u8, 0u8);
    #[max_iterations(8)]
    while let Phase::Rolling = phase {
        let roll = rolls[turn as usize];
        turn += 1;
        if roll == 1 {
            total = 0;
            phase = Phase::Done;
        } else {
            total += roll;
            if total >= 20 || turn == 8 {
                phase = Phase::Done;
            }
        }
    }
    (total, turn)
}

#[test]
fn while_let_on_an_enum() {
    let compiled = press_luck_circuit();
    assert_eq!(compiled.verify(20_000).unwrap().skipped, 0);
    assert_eq!(compiled.eval(&([6, 6, 2, 1, 5, 5, 5, 5],)).unwrap(), (0, 4));
    assert_eq!(compiled.eval(&([6, 6, 6, 3, 5, 5, 5, 5],)).unwrap(), (21, 4));
}

/// Bubble sort until a pass swaps nothing: a `for` inside a `while`. (`slice::swap` isn't supported in
/// `#[handwired]` code, so the swap is written out.)
#[handwired]
#[allow(clippy::manual_swap)]
fn sort(values: [u8; 6]) -> ([u8; 6], u8) {
    let (mut values, mut swapped, mut passes) = (values, true, 0u8);
    #[max_iterations(6)]
    while swapped {
        swapped = false;
        passes += 1;
        for i in 0..5 {
            if values[i] > values[i + 1] {
                let low = values[i + 1];
                values[i + 1] = values[i];
                values[i] = low;
                swapped = true;
            }
        }
    }
    (values, passes)
}

#[test]
fn nested_loops() {
    let compiled = sort_circuit();
    assert_eq!(compiled.verify(20_000).unwrap().skipped, 0);
    assert_eq!(compiled.eval(&([9, 3, 7, 1, 8, 2],)).unwrap(), ([1, 2, 3, 7, 8, 9], 5));
}

/// `continue` inside a `while`, and a `while` whose condition doesn't depend on the inputs, which leaves for real.
#[handwired]
fn skip_thirds(#[range(0, 30)] n: u8) -> (u16, u32) {
    let (mut i, mut total) = (0u8, 0u16);
    #[max_iterations(30)]
    while i < n {
        i += 1;
        if i % 3 == 0 {
            continue;
        }
        total += i as u16;
    }
    let mut power = 1u32;
    #[max_iterations(10)]
    while power < 1000 {
        power *= 3;
    }
    (total, power)
}

#[test]
fn continue_and_native_conditions() {
    let compiled = skip_thirds_circuit();
    for n in 0..=30u8 {
        assert_eq!(compiled.eval(&(n,)).unwrap(), skip_thirds(n), "{n}");
    }
}

/// The same gcd with too small a limit.
#[handwired]
fn short_gcd(#[range(1, 255)] a: u8, #[range(1, 255)] b: u8) -> u8 {
    let (mut a, mut b) = (a, b);
    #[max_iterations(4)]
    while a != b {
        if a > b {
            a -= b;
        } else {
            b -= a;
        }
    }
    a
}

#[test]
fn verify_reports_inputs_past_the_limit() {
    match short_gcd_circuit().verify(2_000) {
        Err(VerifyError::IterationLimit { message, .. }) => {
            assert!(message.contains("more times than its #[max_iterations(4)]"), "{message}");
        }
        other => panic!("expected an iteration limit failure, got {other:?}"),
    }
    // Within the limit, the function runs as written.
    assert_eq!(short_gcd(12, 18), 6);
}

#[test]
#[should_panic(expected = "#[max_iterations(4)]")]
fn the_function_panics_past_its_limit() {
    short_gcd(255, 1);
}
