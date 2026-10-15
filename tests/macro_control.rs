//! `#[handwired]` on arrays, loops, `match`, casts and wrapping arithmetic.
//! Every circuit is verified against the function it came from.

use handwired_rs::handwired;

fn verified<Args, R>(compiled: handwired_rs::dsl::Compiled<Args, R>) -> handwired_rs::dsl::Report
where
    Args: handwired_rs::dsl::Params + Clone + std::fmt::Debug,
    R: handwired_rs::dsl::HwType + PartialEq + std::fmt::Debug,
{
    compiled.verify(10_000).unwrap()
}

/// Dynamic stores, `%` by a constant, and a cast to usize for indexing.
#[handwired]
fn histogram(values: [u8; 6]) -> [u8; 4] {
    let mut counts: [u8; 4] = [0; 4];
    for value in values {
        counts[(value % 4) as usize] += 1;
    }
    counts
}

#[test]
fn histogram_matches() {
    assert_eq!(verified(histogram_circuit()).skipped, 0);
}

const SQUARES: [u16; 16] = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225];

/// A constant table read at an index that depends on the input.
#[handwired]
fn square_of_low_nibble(a: u8) -> u16 {
    SQUARES[(a % 16) as usize] + a as u16
}

#[test]
fn constant_table_lookup() {
    let compiled = square_of_low_nibble_circuit();
    for a in 0..=255u8 {
        assert_eq!(compiled.eval(&(a,)).unwrap(), square_of_low_nibble(a));
    }
}

/// An `if` that both assigns a variable and produces a value, and a `let`
/// without a value that the branches fill in.
#[handwired]
fn mixed(c: bool, #[range(0, 100)] x: u8) -> (u8, u8) {
    let mut x = x;
    let y: u8 = if c {
        x += 1;
        x
    } else {
        0
    };
    let z: u8;
    if x > 50 {
        z = 1;
    } else {
        z = 2;
    }
    (y, x + z)
}

#[test]
fn branches_that_assign_and_produce_values() {
    assert_eq!(verified(mixed_circuit()).skipped, 0);
}

/// A 2-D grid written at a computed row and column.
#[handwired]
fn paint(#[range(0, 2)] row: u8, #[range(0, 3)] column: u8, color: u8) -> [[u8; 3]; 2] {
    let mut grid: [[u8; 3]; 2] = [[0; 3]; 2];
    if row < 2 && column < 3 {
        grid[row as usize][column as usize] = color;
    }
    grid
}

#[test]
fn nested_array_store() {
    assert_eq!(verified(paint_circuit()).skipped, 0);
}

const LIMIT: u8 = 200;

#[handwired]
fn classify(a: u8, b: bool) -> u8 {
    match (a, b) {
        (0, _) => 10,
        (1 | 2 | 3, true) => 11,
        (4..=9, _) => 12,
        (LIMIT, false) => 13,
        (n, true) if n > 100 => n - 100,
        _ => 14,
    }
}

#[test]
fn match_patterns() {
    let compiled = classify_circuit();
    for a in 0..=255u8 {
        for b in [false, true] {
            assert_eq!(compiled.eval(&(a, b)).unwrap(), classify(a, b), "{a} {b}");
        }
    }
}

#[handwired]
fn signed_match(#[range(-5, 5)] a: i8) -> i8 {
    match a {
        -1 => 1,
        -5..=-2 => 2,
        _ => a,
    }
}

#[test]
fn negative_literal_patterns() {
    let compiled = signed_match_circuit();
    for a in -5..=5i8 {
        assert_eq!(compiled.eval(&(a,)).unwrap(), signed_match(a));
    }
}

#[handwired]
fn wrapping(a: u8, b: u8, c: u16) -> (u8, u8, u8, u8) {
    (a.wrapping_add(b), a.wrapping_sub(b), a.saturating_sub(b), c as u8)
}

#[test]
fn wrapping_saturating_and_narrowing() {
    assert_eq!(verified(wrapping_circuit()).skipped, 0);
}

#[handwired]
fn truncate(#[range(-300, 300)] x: f32) -> (i32, u8, i8) {
    (x as i32, x as u8, x as i8)
}

#[test]
fn float_to_integer_casts() {
    let compiled = truncate_circuit();
    for x in [-300.0f32, -129.0, -128.5, -3.5, -2.5, -0.5, -0.0, 0.0, 0.25, 2.5, 3.5, 127.5, 255.9, 256.0, 300.0] {
        assert_eq!(compiled.eval(&(x,)).unwrap(), truncate(x), "{x}");
    }
    assert_eq!(verified(compiled).skipped, 0);
}

/// `break` under a native condition still works: the loop unrolls at build time.
#[handwired]
fn first_three_sum(values: [u8; 8]) -> u16 {
    let mut total: u16 = 0;
    for (index, value) in values.into_iter().enumerate() {
        if index == 3 {
            break;
        }
        total += value as u16;
    }
    total
}

#[test]
fn static_break() {
    assert_eq!(verified(first_three_sum_circuit()).skipped, 0);
}

/// An early `return` under a condition that depends on the inputs.
#[handwired]
fn early_return(a: u8) -> u8 {
    if a > 10 {
        return 1;
    }
    2
}

#[test]
fn dynamic_return() {
    let compiled = early_return_circuit();
    for a in 0..=255u8 {
        assert_eq!(compiled.eval(&(a,)).unwrap(), early_return(a));
    }
}

#[handwired]
fn calls_another(a: u8, b: u8) -> u8 {
    let picked = pick_larger(a, b);
    picked / 2
}

#[handwired]
fn pick_larger(a: u8, b: u8) -> u8 {
    a.max(b)
}

#[test]
fn calls_between_handwired_functions() {
    assert_eq!(verified(calls_another_circuit()).skipped, 0);
}
