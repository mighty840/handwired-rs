//! Variants named without their enum, after `use Enum::*`: in patterns, and as values wherever the enum is
//! expected, which is where the circuit form takes its enum from. Every circuit is verified against the function it
//! came from.

#![deny(missing_docs)]

use handwired_rs::handwired;

/// What a unit does on its turn.
#[derive(Clone, Copy, Debug, PartialEq, handwired_rs::dsl::HwType)]
pub enum Act {
    /// Nothing.
    Wait,
    /// Damage dealt.
    Hit(#[range(0, 9)] u8),
    /// Health restored, and whether it also cleanses.
    Heal(#[range(0, 5)] u8, bool),
    /// A step across.
    Step {
        /// Left or right.
        #[range(-1, 1)]
        dx: i8,
    },
}

/// Another enum with a variant of the same name, so the name alone doesn't say which.
#[derive(Clone, Copy, Debug, PartialEq, handwired_rs::dsl::HwType)]
pub enum Roll {
    /// A hit and whether it's critical.
    Hit(#[range(0, 3)] u8, bool),
    /// A miss.
    Miss,
}

/// A plan for one unit.
#[derive(Clone, Copy, Debug, PartialEq, handwired_rs::dsl::HwType)]
pub struct Plan {
    /// Health.
    #[range(0, 9)]
    pub hp: u8,
    /// What it does.
    pub act: Act,
}

use Act::*;

/// Patterns and values without the enum's name: the function's result, `if` and `match` branches beside qualified
/// variants, an assignment, an array, and a tuple.
#[handwired]
pub fn react(#[range(0, 9)] hp: u8, act: Act) -> (Act, [Act; 2], u8) {
    let mut next = Wait;
    if hp < 3 {
        next = Heal(5, true);
    }
    let reply = match act {
        Wait => Hit(1),
        Hit(d) if d > hp => Act::Heal(hp, false),
        Hit(d) => Hit(d / 2),
        Heal(amount, _) => Hit(amount),
        Step { dx } => Act::Step { dx: -dx },
    };
    let left = match act {
        Hit(d) => hp.saturating_sub(d),
        Heal(amount, _) => (hp + amount).min(9),
        _ => hp,
    };
    (reply, [next, Hit(if hp > 4 { 2 } else { 4 })], left)
}

#[test]
fn bare_variants_in_patterns_branches_and_assignments() {
    let compiled = react_circuit();
    assert_eq!(compiled.verify(20_000).unwrap().skipped, 0);
    assert_eq!(compiled.eval(&(2, Hit(7))).unwrap(), (Heal(2, false), [Heal(5, true), Hit(4)], 0));
    assert_eq!(compiled.eval(&(8, Step { dx: 1 })).unwrap(), (Step { dx: -1 }, [Wait, Hit(2)], 8));
}

/// The halved hit, as a function another one calls with a bare variant.
#[handwired]
pub fn weaken(act: Act) -> Act {
    match act {
        Hit(d) => Hit(d / 2),
        other => other,
    }
}

/// A struct field, an argument, comparisons on either side, an annotated `let` of an `Option`, and an enum with a
/// variant of the same name, named with its enum.
#[handwired]
pub fn plan(#[range(0, 9)] hp: u8, act: Act, critical: bool) -> (Plan, Act, Option<Act>, bool, bool, Roll) {
    let plan = Plan { hp, act: Hit(4) };
    let weaker = weaken(Hit(hp));
    let maybe: Option<Act> = if hp > 4 { Some(Hit(1)) } else { None };
    let roll = if critical { Roll::Hit(3, true) } else { Roll::Miss };
    (plan, weaker, maybe, act == Wait, Hit(3) == act, roll)
}

#[test]
fn bare_variants_take_their_enum_from_fields_arguments_and_comparisons() {
    let compiled = plan_circuit();
    assert_eq!(compiled.verify(20_000).unwrap().skipped, 0);
    let (plan, weaker, maybe, waits, hits_three, roll) = compiled.eval(&(6, Hit(3), true)).unwrap();
    assert_eq!(
        (plan, weaker, maybe, waits, hits_three),
        (Plan { hp: 6, act: Hit(4) }, Hit(3), Some(Hit(1)), false, true)
    );
    assert_eq!(roll, Roll::Hit(3, true));
}

/// Both branches bare, the function's result naming the enum.
#[handwired]
pub fn pick(choice: bool) -> Act {
    if choice { Hit(1) } else { Heal(2, true) }
}

/// Every arm bare.
#[handwired]
pub fn pick_arm(#[range(0, 9)] n: u8) -> Act {
    match n {
        0 => Hit(1),
        1 => Heal(n, false),
        _ => Wait,
    }
}

#[test]
fn both_branches_bare() {
    assert_eq!(pick_circuit().verify(100).unwrap().skipped, 0);
    assert_eq!(pick_arm_circuit().verify(100).unwrap().skipped, 0);
}
