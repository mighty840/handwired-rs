//! `Result<T, E>` and `?`: a bit for whether it's `Err`, then the `Ok` value's cells and the error's, each 0 where
//! the result is the other. Every circuit is verified against the function it came from.

#![deny(missing_docs)]

use handwired_rs::dsl::HwType;
use handwired_rs::handwired;

/// Why a move was refused.
#[derive(Clone, Copy, Debug, PartialEq, HwType)]
pub enum Fault {
    /// Not enough health.
    Weak,
    /// Off the board, by how far.
    Off(#[range(1, 20)] u8),
}

/// A step along a 10-cell track, refused off the ends.
#[handwired]
pub fn step(#[range(0, 9)] at: u8, #[range(-3, 3)] by: i8) -> Result<u8, Fault> {
    let next = at as i8 + by;
    if next < 0 {
        Err(Fault::Off((-next) as u8))
    } else if next > 9 {
        Err(Fault::Off((next - 9) as u8))
    } else {
        Ok(next as u8)
    }
}

#[test]
fn results_are_built_and_returned() {
    let compiled = step_circuit();
    assert_eq!(compiled.verify(2_000).unwrap().skipped, 0);
    assert_eq!(compiled.eval(&(2, 3)).unwrap(), Ok(5));
    assert_eq!(compiled.eval(&(1, -3)).unwrap(), Err(Fault::Off(2)));
    assert_eq!(<Result<u8, Fault> as HwType>::CELLS, 4);
}

/// Two steps, each with `?`, one of them only under an input condition, and a check that fails with its own error.
#[handwired]
pub fn walk(
    #[range(0, 9)] at: u8,
    #[range(-3, 3)] first: i8,
    #[range(-3, 3)] second: i8,
    twice: bool,
    #[range(0, 50)] hp: u8,
) -> Result<u8, Fault> {
    let mut at = step(at, first)?;
    if twice {
        at = step(at, second)?;
    }
    if hp < 5 {
        return Err(Fault::Weak);
    }
    Ok(at * 2)
}

#[test]
fn question_marks_leave_early() {
    let compiled = walk_circuit();
    assert_eq!(compiled.verify(10_000).unwrap().skipped, 0);
    assert_eq!(compiled.eval(&(5, 3, 3, true, 20)).unwrap(), Err(Fault::Off(2)));
    assert_eq!(compiled.eval(&(5, 3, 3, false, 20)).unwrap(), Ok(16));
    assert_eq!(compiled.eval(&(0, -1, 3, true, 20)).unwrap(), Err(Fault::Off(1)));
    assert_eq!(compiled.eval(&(5, 0, 0, true, 2)).unwrap(), Err(Fault::Weak));
}

/// `?` on an Option in a function that returns one.
#[handwired]
pub fn second_above(values: [u8; 4], limit: u8) -> Option<u16> {
    let mut found: Option<u8> = None;
    let mut seen = 0u8;
    for value in values {
        if value > limit {
            seen += 1;
            if seen == 2 {
                found = Some(value);
            }
        }
    }
    let value = found?;
    Some(value as u16 * 3)
}

#[test]
fn question_marks_work_on_options() {
    let compiled = second_above_circuit();
    assert_eq!(compiled.verify(5_000).unwrap().skipped, 0);
    assert_eq!(compiled.eval(&([9, 1, 8, 7], 5)).unwrap(), Some(24));
    assert_eq!(compiled.eval(&([9, 1, 2, 3], 5)).unwrap(), None);
}

/// Matching, `if let`, methods, `==`, and `Ok`/`Err` whose type comes from a field or a later use.
#[handwired]
#[allow(clippy::type_complexity)]
pub fn report(
    #[ranges([(0, 1), (0, 40), (0, 1), (1, 20)])] outcome: Result<u8, Fault>,
    other: Result<u8, Fault>,
    near: bool,
) -> (u8, bool, bool, Option<u8>, Option<Fault>, u8, Result<u8, Fault>) {
    let score = match outcome {
        Ok(v) if v > 30 => 100,
        Ok(v) => v,
        Err(Fault::Weak) => 1,
        Err(Fault::Off(by)) => 50 - by,
    };
    let bonus = if let Err(Fault::Off(by)) = other { by } else { 0 };
    let fallback: Result<u8, Fault> = if near { Ok(4) } else { Err(Fault::Weak) };
    (
        score,
        outcome == other,
        outcome.is_ok() && other.is_err(),
        outcome.ok(),
        other.err(),
        outcome.unwrap_or(7) + bonus,
        fallback,
    )
}

#[test]
fn results_match_and_compare() {
    let compiled = report_circuit();
    assert_eq!(compiled.verify(10_000).unwrap().skipped, 0);
    let cases = [
        (Ok(35), Ok(35), true),
        (Ok(3), Err(Fault::Off(4)), false),
        (Err(Fault::Weak), Err(Fault::Weak), true),
        (Err(Fault::Off(2)), Err(Fault::Off(3)), false),
        (Err(Fault::Off(2)), Ok(0), true),
    ];
    for (outcome, other, near) in cases {
        assert_eq!(compiled.eval(&(outcome, other, near)).unwrap(), report(outcome, other, near), "{outcome:?}");
    }
}

/// A unit's last action, a Result as a struct field.
#[derive(Clone, Copy, Debug, PartialEq, HwType)]
pub struct Log {
    /// Where it got to, or why not.
    pub last: Result<u8, Fault>,
    /// How many steps were tried.
    #[range(0, 10)]
    pub tries: u8,
}

/// `Ok`/`Err` of plain literals in a field, and in an `if` there.
#[handwired]
pub fn record(log: Log, #[range(0, 9)] at: u8, blocked: bool) -> Log {
    Log { last: if blocked { Err(Fault::Off(1)) } else { Ok(at) }, tries: log.tries + 1 }
}

#[test]
fn results_are_struct_fields() {
    let compiled = record_circuit();
    assert_eq!(compiled.verify(2_000).unwrap().skipped, 0);
    let log = Log { last: Ok(3), tries: 2 };
    assert_eq!(compiled.eval(&(log, 4, true)).unwrap(), Log { last: Err(Fault::Off(1)), tries: 3 });
}

/// `?` inside a `for` body and a `while let` with a limit: the first refused step ends the function.
#[handwired]
pub fn route(
    #[range(0, 9)] start: u8,
    #[range(-3, 3)] moves: [i8; 4],
    #[range(-3, 3)] plan: Option<i8>,
) -> Result<u8, Fault> {
    let mut at = start;
    for by in moves {
        at = step(at, by)?;
    }
    let mut extra = plan;
    #[max_iterations(4)]
    while let Some(by) = extra {
        at = step(at, by)?;
        extra = if by > 0 { Some(by - 1) } else { None };
    }
    Ok(at)
}

#[test]
fn question_marks_leave_loops() {
    let compiled = route_circuit();
    assert_eq!(compiled.verify(10_000).unwrap().skipped, 0);
    for (start, moves, plan) in [
        (0, [3, 3, 3, -3], None),
        (0, [3, 3, 3, 3], None),
        (5, [-3, -3, 3, 3], Some(2)),
        (5, [0, 0, 0, 0], Some(1)),
        (8, [0, 0, 0, 0], Some(2)),
        (9, [-3, 0, 0, 0], Some(-1)),
    ] {
        assert_eq!(compiled.eval(&(start, moves, plan)).unwrap(), route(start, moves, plan), "{start} {moves:?}");
    }
}
