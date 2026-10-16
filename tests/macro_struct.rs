//! `#[derive(HwType)]`: structs as `#[handwired]` parameters, locals and
//! results. Every circuit is verified against the function it came from.

#![deny(missing_docs)]

use handwired_rs::dsl::HwType;
use handwired_rs::handwired;

/// The state of a two-player game of Pig: banked scores, whose turn it is, and the points at stake this turn.
#[derive(Clone, Copy, Debug, PartialEq, HwType)]
pub struct Pig {
    /// Each player's banked score.
    #[range(0, 100)]
    pub scores: [u8; 2],
    /// 0 or 1.
    #[range(0, 1)]
    pub turn: u8,
    /// The points rolled this turn, lost on a 1.
    #[range(0, 30)]
    pub pot: u8,
}

/// A new game, as a struct literal.
#[handwired]
pub fn new_game(#[range(0, 1)] first: u8) -> Pig {
    Pig { scores: [0, 0], turn: first, pot: 0 }
}

/// One move: hold banks the pot and passes the turn; a roll of 1 loses it; any other roll adds to it.
#[handwired]
pub fn pig_step(state: Pig, #[range(1, 6)] roll: u8, hold: bool) -> Pig {
    let mut next = state;
    if hold {
        next.scores[state.turn as usize] += state.pot;
        next.pot = 0;
        next.turn = 1 - state.turn;
    } else if roll == 1 {
        next.pot = 0;
        next.turn = 1 - state.turn;
    } else {
        next.pot += roll;
    }
    next
}

#[test]
fn a_game_state_is_a_parameter_a_local_and_a_result() {
    let compiled = new_game_circuit();
    assert_eq!(compiled.eval(&(1,)).unwrap(), Pig { scores: [0, 0], turn: 1, pot: 0 });
    let compiled = pig_step_circuit();
    assert_eq!(compiled.verify(20_000).unwrap().skipped, 0);
    // A whole game, the state carried from move to move through the circuit.
    let mut state = Pig { scores: [0, 0], turn: 0, pot: 0 };
    let moves = [(4, false), (5, false), (3, true), (1, false), (6, false), (2, true), (6, false), (6, true)];
    for (roll, hold) in moves {
        let next = compiled.eval(&(state, roll, hold)).unwrap();
        assert_eq!(next, pig_step(state, roll, hold));
        state = next;
    }
    // Player 0 banks 4 + 5 = 9, player 1 rolls a 1, player 0 banks 6 more, player 1 banks 6 (holds ignore their roll).
    assert_eq!(state, Pig { scores: [15, 6], turn: 0, pot: 0 });
    // The fields' ranges are declared cell by cell: scores, turn, pot, then roll and hold.
    let ranges: Vec<(f64, f64)> = (0..4)
        .map(|cell| {
            let range = compiled.circuit.range(compiled.circuit.source(cell));
            (range.min, range.max)
        })
        .collect();
    assert_eq!(ranges, [(0.0, 100.0), (0.0, 100.0), (0.0, 1.0), (0.0, 30.0)]);
    assert_eq!(<Pig as HwType>::CELLS, 4);
}

/// A unit's damage and health.
#[derive(Clone, Copy, Debug, PartialEq, HwType)]
pub struct Unit {
    /// Damage per hit.
    #[range(1, 9)]
    pub damage: u8,
    /// Health left.
    #[range(0, 40)]
    pub health: u8,
}

/// Three units, a nested struct.
#[derive(Clone, Copy, Debug, PartialEq, HwType)]
pub struct Board {
    /// The units, in board order.
    pub units: [Unit; 3],
    /// Attacks made so far.
    #[range(0, 10)]
    pub attacks: u8,
}

/// Unit `attacker` hits unit `target`, both chosen by the inputs: struct elements read and written at computed
/// indices.
#[handwired]
pub fn attack(board: Board, #[range(0, 2)] attacker: u8, #[range(0, 2)] target: u8) -> Board {
    let mut next = board;
    let damage = board.units[attacker as usize].damage;
    let health = board.units[target as usize].health;
    next.units[target as usize].health = health.saturating_sub(damage);
    next.attacks += 1;
    next
}

#[test]
fn nested_structs_and_arrays_of_structs_at_computed_indices() {
    let compiled = attack_circuit();
    assert_eq!(compiled.verify(20_000).unwrap().skipped, 0);
    let board = Board {
        units: [Unit { damage: 3, health: 10 }, Unit { damage: 7, health: 5 }, Unit { damage: 1, health: 40 }],
        attacks: 0,
    };
    let records: Vec<(Board, u8, u8)> = (0..9u8).map(|pair| (board, pair / 3, pair % 3)).collect();
    for (record, result) in records.iter().zip(compiled.run_batch(&records).unwrap()) {
        assert_eq!(result, attack(record.0, record.1, record.2));
    }
    assert_eq!(compiled.eval(&(board, 1, 0)).unwrap().units[0].health, 3);
}

/// Per-scalar ranges on a parameter override the fields' own, cell by cell.
#[handwired]
pub fn total_health(#[ranges([(1, 2), (0, 7), (1, 2), (0, 7), (1, 2), (0, 7), (0, 0)])] board: Board) -> u16 {
    board.units[0].health as u16 + board.units[1].health as u16 + board.units[2].health as u16
}

#[test]
fn parameter_ranges_override_field_ranges() {
    let compiled = total_health_circuit();
    let health = compiled.circuit.range(compiled.circuit.source(1));
    assert_eq!((health.min, health.max), (0.0, 7.0));
    assert_eq!(compiled.verify(2_000).unwrap().skipped, 0);
}

/// Structs as an `if`'s value, in struct update syntax, in an annotated local, and in an early return.
#[handwired]
pub fn resolve(state: Pig, #[range(1, 6)] roll: u8, hold: bool) -> Pig {
    if state.scores[0] >= 100 {
        return Pig { pot: 0, ..state };
    }
    let busted: Pig = Pig { pot: 0, turn: 1 - state.turn, ..state };
    let after = if roll == 1 { busted } else { Pig { pot: state.pot + roll, ..state } };
    if hold { Pig { pot: 0, turn: 1 - state.turn, ..state } } else { after }
}

#[test]
fn structs_in_if_values_updates_and_early_returns() {
    let compiled = resolve_circuit();
    assert_eq!(compiled.verify(20_000).unwrap().skipped, 0);
}

/// A GPU-free check that a struct result decodes from a batch.
#[test]
fn struct_results_decode_from_batches() {
    let compiled = pig_step_circuit();
    let records: Vec<(Pig, u8, bool)> = (0..300u32)
        .map(|n| {
            (
                Pig { scores: [(n % 90) as u8, (n * 7 % 90) as u8], turn: (n % 2) as u8, pot: (n % 25) as u8 },
                (1 + n % 6) as u8,
                n % 3 == 0,
            )
        })
        .collect();
    for (record, result) in records.iter().zip(compiled.run_batch(&records).unwrap()) {
        assert_eq!(result, pig_step(record.0, record.1, record.2));
    }
}
