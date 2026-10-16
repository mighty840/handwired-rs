//! `return`, `break` and `continue` under conditions that depend on the inputs.
//! Every circuit is verified against the function it came from.

// The functions are written the way people write them, to exercise the macro, not the way clippy prefers.
#![allow(
    clippy::needless_return,
    clippy::needless_range_loop,
    clippy::needless_late_init,
    clippy::manual_range_patterns
)]

use handwired_rs::handwired;

fn check<Args, R>(compiled: handwired_rs::dsl::Compiled<Args, R>)
where
    Args: handwired_rs::dsl::Params + Clone + std::fmt::Debug + Sync,
    R: handwired_rs::dsl::HwType + PartialEq + std::fmt::Debug,
{
    let report = compiled.verify(20_000).unwrap();
    assert!(report.checked > 10_000, "{report:?}");
}

/// A linear search that returns from inside the loop.
#[handwired]
fn find(values: [u8; 8], target: u8) -> u8 {
    for (index, value) in values.into_iter().enumerate() {
        if value == target {
            return index as u8;
        }
    }
    255
}

#[test]
fn return_from_a_loop() {
    let compiled = find_circuit();
    check(find_circuit());
    // Random arrays rarely contain the target, so also plant it.
    for position in 0..8 {
        let mut values = [7u8, 3, 9, 3, 0, 255, 128, 3];
        values[position] = 42;
        assert_eq!(compiled.eval(&(values, 42)).unwrap(), position as u8);
    }
    assert_eq!(compiled.eval(&([3, 1, 3, 1, 3, 1, 3, 1], 3)).unwrap(), 0);
    assert_eq!(compiled.eval(&([1, 1, 1, 1, 1, 1, 1, 3], 3)).unwrap(), 7);
}

/// Sum until a zero, skipping odd values.
#[handwired]
fn sum_until_zero(#[range(0, 9)] values: [u8; 10]) -> u16 {
    let mut total: u16 = 0;
    for value in values {
        if value == 0 {
            break;
        }
        if value % 2 == 1 {
            continue;
        }
        total += value as u16;
    }
    total
}

#[test]
fn break_and_continue() {
    check(sum_until_zero_circuit());
}

/// Writes to an array stop at a break, and a continue skips one write.
#[handwired]
fn prefix_marks(#[range(0, 3)] values: [u8; 6]) -> [bool; 6] {
    let mut marks: [bool; 6] = [false; 6];
    for (index, value) in values.into_iter().enumerate() {
        if value == 3 {
            break;
        }
        if value == 1 {
            continue;
        }
        marks[index] = true;
    }
    marks
}

#[test]
fn masked_array_writes() {
    check(prefix_marks_circuit());
}

/// A break in an inner loop stops only that loop; a return leaves both.
#[handwired]
fn grid_search(#[range(0, 4)] grid: [[u8; 4]; 3]) -> (u8, u8) {
    let mut zeros: u8 = 0;
    for (row, cells) in grid.into_iter().enumerate() {
        for cell in cells {
            if cell == 4 {
                return (row as u8, zeros);
            }
            if cell == 3 {
                break;
            }
            if cell == 0 {
                zeros += 1;
            }
        }
    }
    (255, zeros)
}

#[test]
fn nested_loops() {
    check(grid_search_circuit());
}

/// `return` as a value: a match arm and an `if` branch that leave early.
#[handwired]
fn returns_as_values(a: u8, b: bool) -> u8 {
    let doubled: u8 = match a {
        0 => return 5,
        1..=100 => a * 2,
        _ => return 6,
    };
    let half = if b { return doubled } else { doubled / 2 };
    half + 1
}

#[test]
fn diverging_branch_values() {
    check(returns_as_values_circuit());
}

/// A function that ends with `return x;`.
#[handwired]
fn trailing_return(a: u8) -> u8 {
    let b = a / 3;
    if b > 40 {
        return b;
    }
    return b + 1;
}

#[test]
fn return_statement_at_the_end() {
    let compiled = trailing_return_circuit();
    for a in 0..=255u8 {
        assert_eq!(compiled.eval(&(a,)).unwrap(), trailing_return(a));
    }
}

/// A break under a native condition still leaves the unrolled loop and costs nothing.
#[handwired]
fn static_break_only(values: [u8; 8]) -> u16 {
    let mut total: u16 = 0;
    for (index, value) in values.into_iter().enumerate() {
        if index == 3 {
            break;
        }
        total += value as u16;
    }
    total
}

#[handwired]
fn no_break(values: [u8; 3]) -> u16 {
    let mut total: u16 = 0;
    for value in values {
        total += value as u16;
    }
    total
}

#[test]
fn static_breaks_cost_nothing() {
    let with_break = static_break_only_circuit();
    let without = no_break_circuit();
    check(static_break_only_circuit());
    assert_eq!(with_break.program.live_operation_count(), without.program.live_operation_count());
}

const TABLE: [u8; 4] = [10, 20, 30, 40];

/// A `return` under a native condition stops the trace, like Rust: past k = 4,
/// `TABLE[k]` would be out of bounds while the circuit is built.
#[handwired]
fn static_return_guards_index(a: u8) -> u8 {
    for k in 0..8 {
        if k == 4 {
            return 0;
        }
        if TABLE[k] == a {
            return k as u8 + 1;
        }
    }
    9
}

#[test]
fn static_return_stops_the_trace() {
    let compiled = static_return_guards_index_circuit();
    for a in 0..=255u8 {
        assert_eq!(compiled.eval(&(a,)).unwrap(), static_return_guards_index(a));
    }
}

/// A `continue` before a `return` in the same iteration: paths that continued must not return there.
#[handwired]
fn continue_then_return(#[range(0, 8)] values: [u8; 6]) -> u8 {
    let mut seen: u8 = 0;
    for value in values {
        if value == 0 {
            continue;
        }
        if value == 7 {
            return seen;
        }
        seen += 1;
    }
    100 + seen
}

#[test]
fn continue_before_return() {
    check(continue_then_return_circuit());
}
