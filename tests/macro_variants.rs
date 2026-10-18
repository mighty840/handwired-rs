//! `Option<T>` and `#[derive(HwType)]` enums whose variants carry data: which
//! variant in one cell, then every variant's fields, which hold 0 where the
//! value is another variant. Every circuit is verified against the function it
//! came from.

#![deny(missing_docs)]

use handwired_rs::dsl::HwType;
use handwired_rs::handwired;

/// The first value above a limit, if any.
#[handwired]
pub fn first_above(values: [u8; 6], limit: u8) -> Option<u8> {
    let mut found: Option<u8> = None;
    for value in values {
        if found.is_none() && value > limit {
            found = Some(value);
        }
    }
    found
}

#[test]
fn an_option_is_a_local_and_a_result() {
    let compiled = first_above_circuit();
    assert_eq!(compiled.verify(10_000).unwrap().skipped, 0);
    assert_eq!(compiled.eval(&([1, 9, 3, 12, 0, 0], 5)).unwrap(), Some(9));
    assert_eq!(compiled.eval(&([1, 2, 3, 4, 5, 5], 5)).unwrap(), None);
    assert_eq!(<Option<u8> as HwType>::CELLS, 2);
}

/// Points for an optional roll: literal and binding patterns, a guard, and `None`.
#[handwired]
pub fn bonus(#[range(1, 6)] roll: Option<u8>, streak: u8) -> u16 {
    match roll {
        Some(6) => 100,
        Some(r) if r >= 4 => r as u16 * 10,
        Some(_) => 1,
        None => streak as u16,
    }
}

#[test]
fn options_match_with_patterns() {
    let compiled = bonus_circuit();
    assert_eq!(compiled.verify(10_000).unwrap().skipped, 0);
    assert_eq!(compiled.eval(&(Some(6), 3)).unwrap(), 100);
    assert_eq!(compiled.eval(&(Some(5), 3)).unwrap(), 50);
    assert_eq!(compiled.eval(&(None, 3)).unwrap(), 3);
    // The value's cell is declared 1..6, with 0 for `None`.
    let range = compiled.circuit.range(compiled.circuit.source(1));
    assert_eq!((range.min, range.max), (0.0, 6.0));
    // Out of the declared range, the encoded value is refused.
    assert!(compiled.eval(&(Some(7), 3)).is_err());
}

/// `if let`, `unwrap_or`, `==` between Options and with `None`, and `Some` of a circuit value.
#[handwired]
#[allow(clippy::manual_map, clippy::partialeq_to_none)]
pub fn pick(a: Option<u8>, b: Option<u8>) -> (u8, bool, bool, Option<u8>) {
    let both = if let Some(x) = a { if let Some(y) = b { Some(x.max(y)) } else { None } } else { None };
    (a.unwrap_or(7), a == b, b == None, both)
}

#[test]
fn option_methods_and_equality() {
    let compiled = pick_circuit();
    assert_eq!(compiled.verify(10_000).unwrap().skipped, 0);
    for (a, b) in [(Some(3), Some(3)), (Some(3), Some(4)), (None, None), (None, Some(0)), (Some(0), None)] {
        assert_eq!(compiled.eval(&(a, b)).unwrap(), pick(a, b));
    }
}

/// What a unit does on its turn.
#[derive(Clone, Copy, Debug, PartialEq, HwType)]
pub enum Action {
    /// Nothing.
    Wait,
    /// One step.
    Move {
        /// Across.
        #[range(-1, 1)]
        dx: i8,
        /// Down.
        #[range(-1, 1)]
        dy: i8,
    },
    /// Damage dealt.
    Attack(#[range(1, 6)] u8),
    /// Health restored, or all of it.
    Heal(#[range(0, 20)] u8, bool),
}

/// A unit on a 10 by 10 board.
#[derive(Clone, Copy, Debug, PartialEq, HwType)]
pub struct Unit {
    /// Column.
    #[range(0, 9)]
    pub x: i8,
    /// Row.
    #[range(0, 9)]
    pub y: i8,
    /// Health.
    #[range(0, 50)]
    pub hp: u8,
}

/// One action applied: a match over every kind of variant, with fields bound by name and by position.
#[handwired]
pub fn apply(unit: Unit, action: Action) -> (Unit, Option<u8>) {
    let mut next = unit;
    let mut dealt: Option<u8> = None;
    match action {
        Action::Wait => {}
        Action::Move { dx, dy } => {
            next.x = (unit.x + dx).clamp(0, 9);
            next.y = (unit.y + dy).clamp(0, 9);
        }
        Action::Attack(damage) => dealt = Some(damage * 2),
        Action::Heal(amount, full) => {
            next.hp = if full { 50 } else { (unit.hp + amount).min(50) };
        }
    }
    (next, dealt)
}

#[test]
fn enums_with_data_match_by_variant() {
    let compiled = apply_circuit();
    assert_eq!(compiled.verify(20_000).unwrap().skipped, 0);
    let unit = Unit { x: 9, y: 0, hp: 40 };
    let cases = [
        (Action::Wait, (unit, None)),
        (Action::Move { dx: 1, dy: -1 }, (unit, None)),
        (Action::Move { dx: -1, dy: 1 }, (Unit { x: 8, y: 1, hp: 40 }, None)),
        (Action::Attack(5), (unit, Some(10))),
        (Action::Heal(20, false), (Unit { hp: 50, ..unit }, None)),
        (Action::Heal(3, false), (Unit { hp: 43, ..unit }, None)),
    ];
    for (action, expected) in cases {
        assert_eq!(compiled.eval(&(unit, action)).unwrap(), expected, "{action:?}");
    }
    // The tag, then Move's two fields, Attack's one, and Heal's two.
    assert_eq!(<Action as HwType>::CELLS, 6);
    let ranges: Vec<(f64, f64)> = (3..9)
        .map(|cell| {
            let range = compiled.circuit.range(compiled.circuit.source(cell));
            (range.min, range.max)
        })
        .collect();
    assert_eq!(ranges, [(0.0, 3.0), (-1.0, 1.0), (-1.0, 1.0), (0.0, 6.0), (0.0, 20.0), (0.0, 1.0)]);
}

/// A unit's choice, built from each kind of variant.
#[handwired]
pub fn choose(#[range(0, 50)] hp: u8, #[range(-3, 3)] dx: i8, enemy_near: bool) -> Action {
    if hp < 10 {
        Action::Heal(10, hp < 3)
    } else if enemy_near {
        Action::Attack(if hp > 30 { 6 } else { 3 })
    } else if dx != 0 {
        Action::Move { dx: dx.clamp(-1, 1), dy: 0 }
    } else {
        Action::Wait
    }
}

#[test]
fn variants_are_built_from_circuit_values() {
    let compiled = choose_circuit();
    assert_eq!(compiled.verify(10_000).unwrap().skipped, 0);
    assert_eq!(compiled.eval(&(2, 0, true)).unwrap(), Action::Heal(10, true));
    assert_eq!(compiled.eval(&(40, 0, true)).unwrap(), Action::Attack(6));
    assert_eq!(compiled.eval(&(40, -3, false)).unwrap(), Action::Move { dx: -1, dy: 0 });
    assert_eq!(compiled.eval(&(40, 0, false)).unwrap(), Action::Wait);
}

/// A queue read at a circuit index, `==` between circuit values and with a constant, and `matches!` with a guard.
#[handwired]
pub fn queue(actions: [Action; 3], #[range(0, 2)] at: usize, other: Action) -> (Action, bool, bool, bool) {
    let current = actions[at];
    (current, current == other, matches!(current, Action::Attack(d) if d > 3), current != Action::Wait)
}

#[test]
fn enums_with_data_compare_and_index() {
    let compiled = queue_circuit();
    assert_eq!(compiled.verify(20_000).unwrap().skipped, 0);
    let actions = [Action::Attack(5), Action::Move { dx: 0, dy: 1 }, Action::Wait];
    assert_eq!(compiled.eval(&(actions, 0, Action::Attack(5))).unwrap(), (Action::Attack(5), true, true, true));
    assert_eq!(compiled.eval(&(actions, 0, Action::Attack(4))).unwrap(), (Action::Attack(5), false, true, true));
    assert_eq!(compiled.eval(&(actions, 2, Action::Wait)).unwrap(), (Action::Wait, true, false, false));
    let moved = Action::Move { dx: 0, dy: 1 };
    assert_eq!(compiled.eval(&(actions, 1, moved)).unwrap(), (moved, true, false, true));
}

/// Options of enums with data, matched through both layers.
#[handwired]
pub fn plan(next: Option<Action>, #[range(0, 50)] hp: u8) -> Option<Action> {
    match next {
        Some(Action::Attack(damage)) if hp < 5 => Some(Action::Heal(damage, false)),
        Some(Action::Move { dx: 0, dy: 0 }) => None,
        Some(action) => Some(action),
        None => None,
    }
}

#[test]
fn options_of_enums_nest() {
    let compiled = plan_circuit();
    assert_eq!(compiled.verify(20_000).unwrap().skipped, 0);
    assert_eq!(compiled.eval(&(Some(Action::Attack(4)), 3)).unwrap(), Some(Action::Heal(4, false)));
    assert_eq!(compiled.eval(&(Some(Action::Move { dx: 0, dy: 0 }), 3)).unwrap(), None);
    assert_eq!(compiled.eval(&(Some(Action::Wait), 3)).unwrap(), Some(Action::Wait));
    assert_eq!(compiled.eval(&(None, 3)).unwrap(), None);
}

/// A unit with its queued action, an enum with data as a struct field.
#[derive(Clone, Copy, Debug, PartialEq, HwType)]
pub struct Turn {
    /// Who acts.
    pub unit: Unit,
    /// What it will do.
    pub action: Action,
}

/// The queued action, carried out, then cleared to `Wait`.
#[handwired]
pub fn take_turn(turn: Turn) -> Turn {
    let (unit, _) = apply(turn.unit, turn.action);
    Turn { unit, action: Action::Wait }
}

#[test]
fn enums_with_data_are_struct_fields() {
    let compiled = take_turn_circuit();
    assert_eq!(compiled.verify(10_000).unwrap().skipped, 0);
    let turn = Turn { unit: Unit { x: 4, y: 4, hp: 10 }, action: Action::Move { dx: 1, dy: 1 } };
    assert_eq!(compiled.eval(&(turn,)).unwrap(), Turn { unit: Unit { x: 5, y: 5, hp: 10 }, action: Action::Wait });
}

/// `#[ranges]` on an Option: one range per cell, whether it's `Some` first.
#[handwired]
pub fn scaled(#[ranges([(0, 1), (10, 20)])] value: Option<u16>) -> u16 {
    match value {
        Some(v) => v * 3,
        None => 0,
    }
}

#[test]
fn option_cells_take_ranges() {
    let compiled = scaled_circuit();
    assert_eq!(compiled.verify(2_000).unwrap().skipped, 0);
    let range = compiled.circuit.range(compiled.circuit.source(1));
    assert_eq!((range.min, range.max), (0.0, 20.0));
    // The circuit is exact over the widened range too.
    assert_eq!(compiled.eval(&(Some(9),)).unwrap(), 27);
    assert_eq!(compiled.eval(&(Some(20),)).unwrap(), 60);
    assert!(compiled.eval(&(Some(21),)).is_err());
}

/// `while let` over an Option that the loop replaces.
#[handwired]
pub fn halvings(start: Option<u8>) -> u8 {
    let mut current: Option<u8> = start;
    let mut steps = 0u8;
    #[max_iterations(5)]
    while let Some(value) = current {
        steps += 1;
        current = if value > 50 { Some(value / 2) } else { None };
    }
    steps
}

#[test]
fn while_let_runs_over_options() {
    let compiled = halvings_circuit();
    assert_eq!(compiled.verify(2_000).unwrap().skipped, 0);
    assert_eq!(compiled.eval(&(Some(255),)).unwrap(), 4);
    assert_eq!(compiled.eval(&(None,)).unwrap(), 0);
}

/// A spot on the board with its marks and an optional bonus.
#[derive(Clone, Copy, Debug, PartialEq, HwType)]
pub struct Spot {
    /// Column and row.
    pub at: (i8, i8),
    /// Two marks.
    pub marks: [u8; 2],
    /// A bonus, if any.
    pub bonus: Option<u8>,
    /// Health.
    #[range(0, 50)]
    pub hp: u8,
}

/// Plain integer literals in `if`s, `match`es, blocks, tuples and arrays inside struct and variant literals take the
/// field's type.
#[handwired]
pub fn place(#[range(0, 3)] kind: u8, near: bool) -> (Spot, Action, Option<u8>, u8) {
    let mut steps = 0u8;
    let size = 2;
    let spot = Spot {
        at: (if near { 1 } else { -1 }, if size > 1 { 2 } else { 3 }),
        marks: [
            match kind {
                0 => 7,
                1 => 8,
                _ => 9,
            },
            if kind > 1 { 200 } else { 100 },
        ],
        bonus: Some(if near {
            steps += 1;
            5
        } else {
            6
        }),
        hp: {
            let base = 10u8;
            if kind == 3 { base * 2 } else { 40 }
        },
    };
    let action = Action::Move { dx: if near { 1 } else { 0 }, dy: if kind == 0 { -1 } else { 1 } };
    let extra: Option<u8> = Some(if kind == 2 { 30 } else { 31 });
    (spot, action, extra, steps)
}

#[test]
fn literals_in_fields_take_the_fields_types() {
    let compiled = place_circuit();
    assert_eq!(compiled.verify(1_000).unwrap().skipped, 0);
    for kind in 0..4 {
        for near in [false, true] {
            assert_eq!(compiled.eval(&(kind, near)).unwrap(), place(kind, near));
        }
    }
}
