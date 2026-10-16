//! `#[derive(HwType)]` on enums whose variants carry no data: one cell holding
//! the discriminant, matched with `match`, `matches!` and `if let`. Every
//! circuit is verified against the function it came from.

#![deny(missing_docs)]

use handwired_rs::dsl::HwType;
use handwired_rs::handwired;

/// Where a turn of Pig stands.
#[derive(Clone, Copy, Debug, PartialEq, HwType)]
pub enum Phase {
    /// The player may roll again.
    Rolling,
    /// The player banked the pot.
    Held,
    /// The player rolled a 1 and lost the pot.
    Busted,
    /// Someone reached the target.
    Won,
}

use Phase::*;

/// A turn of Pig with its phase as an enum, as a struct field.
#[derive(Clone, Copy, Debug, PartialEq, HwType)]
pub struct Turn {
    /// Where the turn stands.
    pub phase: Phase,
    /// The banked score.
    #[range(0, 100)]
    pub score: u8,
    /// The points at stake.
    #[range(0, 30)]
    pub pot: u8,
}

/// One step: match on the phase, set a new phase by path, or-patterns, a guard and a wildcard.
#[handwired]
pub fn turn_step(turn: Turn, #[range(1, 6)] roll: u8, hold: bool) -> Turn {
    let mut next = turn;
    match turn.phase {
        Phase::Rolling if hold => {
            next.score = turn.score + turn.pot;
            next.pot = 0;
            next.phase = if next.score >= 50 { Phase::Won } else { Phase::Held };
        }
        Phase::Rolling => {
            if roll == 1 {
                next.pot = 0;
                next.phase = Phase::Busted;
            } else {
                next.pot = turn.pot + roll;
            }
        }
        Held | Busted => next.phase = Rolling,
        _ => {}
    }
    next
}

#[test]
fn match_on_an_enum_field_steps_a_turn() {
    let compiled = turn_step_circuit();
    assert_eq!(compiled.verify(20_000).unwrap().skipped, 0);
    let mut turn = Turn { phase: Phase::Rolling, score: 40, pot: 0 };
    for (roll, hold) in [(6, false), (5, false), (2, true)] {
        turn = compiled.eval(&(turn, roll, hold)).unwrap();
    }
    assert_eq!(turn, Turn { phase: Phase::Won, score: 51, pot: 0 });
    // The phase is one cell, declared as the discriminants' range.
    assert_eq!(<Turn as HwType>::CELLS, 3);
    let range = compiled.circuit.range(compiled.circuit.source(0));
    assert_eq!((range.min, range.max), (0.0, 3.0));
}

/// Explicit discriminants with gaps, and a derived order.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, HwType)]
pub enum Tier {
    /// Below zero.
    Low = -2,
    /// In between.
    Mid = 5,
    /// Above.
    High = 40,
}

/// Discriminants through `as` (wrapping into u8 for `Low`), ordering comparisons, and an enum result.
#[handwired]
pub fn tier_info(tier: Tier, other: Tier) -> (i32, u8, bool, Tier) {
    let higher = if tier > other { tier } else { other };
    (tier as i32 * 3, tier as u8, tier <= Tier::Mid, higher)
}

#[test]
fn discriminants_casts_and_order_match_rust() {
    let compiled = tier_info_circuit();
    assert_eq!(compiled.verify(5_000).unwrap().skipped, 0);
    assert_eq!(compiled.eval(&(Tier::Low, Tier::High)).unwrap(), (-6, 254, true, Tier::High));
    let range = compiled.circuit.range(compiled.circuit.source(0));
    assert_eq!((range.min, range.max), (-2.0, 40.0));
    // Sampling draws only real variants, each of them.
    let mut rng = handwired_rs::dsl::Rng::new(7);
    let attrs = handwired_rs::dsl::Attrs::default();
    let drawn: Vec<Tier> = (0..60).map(|_| <Tier as HwType>::sample(&mut rng, &attrs)).collect();
    for tier in [Tier::Low, Tier::Mid, Tier::High] {
        assert!(drawn.contains(&tier));
    }
}

/// A tic-tac-toe square.
#[derive(Clone, Copy, Debug, PartialEq, HwType)]
pub enum Cell {
    /// Nobody's.
    Empty,
    /// The first player's.
    X,
    /// The second player's.
    O,
}

const LINES: [[usize; 3]; 8] = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];

/// Place a mark at a computed square if it's empty, then report the winner: arrays of enums read and written through
/// circuit indices, `matches!` with a guard, and `if let`.
#[handwired]
pub fn place(board: [Cell; 9], #[range(0, 8)] square: u8, mark: Cell) -> ([Cell; 9], Cell, i8) {
    let mut next = board;
    if matches!(board[square as usize], Cell::Empty) && mark != Cell::Empty {
        next[square as usize] = mark;
    }
    let mut winner = Cell::Empty;
    for line in LINES {
        let first = next[line[0]];
        if matches!(first, Cell::X | Cell::O if next[line[1]] == first && next[line[2]] == first) {
            winner = first;
        }
    }
    let score: i8 = if let Cell::X = winner {
        1
    } else if let Cell::O = winner {
        -1
    } else {
        0
    };
    (next, winner, score)
}

#[test]
fn enum_arrays_read_and_write_through_circuit_indices() {
    let compiled = place_circuit();
    assert_eq!(compiled.verify(20_000).unwrap().skipped, 0);
    use Cell::{Empty as E, O, X};
    let board = [X, X, E, O, O, E, E, E, E];
    assert_eq!(compiled.eval(&(board, 2, X)).unwrap(), ([X, X, X, O, O, E, E, E, E], X, 1));
    assert_eq!(compiled.eval(&(board, 0, O)).unwrap(), (board, E, 0));
    assert_eq!(compiled.eval(&(board, 5, O)).unwrap(), ([X, X, E, O, O, O, E, E, E], O, -1));
}

/// What comes after each phase, as a native table read at a circuit index.
const AFTER: [Phase; 4] = [Phase::Rolling, Phase::Rolling, Phase::Rolling, Phase::Won];

/// A table of enums at a cast index, `if let` without `else`, and `matches!` in tail position.
#[handwired]
pub fn advance(phase: Phase, #[range(0, 3)] bonus: u8) -> (Phase, u8, bool) {
    let next = AFTER[phase as usize];
    let mut points = bonus;
    if let Phase::Busted = phase {
        points = 0;
    }
    (next, points, finished(next))
}

/// Whether a phase ends the game.
#[handwired]
pub fn finished(phase: Phase) -> bool {
    matches!(phase, Won)
}

#[test]
fn enum_tables_if_let_and_matches() {
    let compiled = advance_circuit();
    assert_eq!(compiled.verify(2_000).unwrap().skipped, 0);
    assert_eq!(compiled.eval(&(Phase::Busted, 3)).unwrap(), (Phase::Rolling, 0, false));
    assert_eq!(compiled.eval(&(Phase::Won, 2)).unwrap(), (Phase::Won, 2, true));
    assert_eq!(finished_circuit().verify(100).unwrap().skipped, 0);
}
