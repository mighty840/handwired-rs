//! `<`, `<=`, `>`, `>=`, `min`, `max` and `clamp` on `Option`, `Result` and enums with data, in the order Rust
//! derives: the variant in declaration order, then its fields in order. Every circuit is verified against the function
//! it came from.

#![deny(missing_docs)]

use handwired_rs::dsl::HwType;
use handwired_rs::handwired;

/// A spot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, HwType)]
pub struct Spot {
    /// Column.
    #[range(-3, 3)]
    pub x: i8,
    /// Row.
    #[range(0, 3)]
    pub y: u8,
}

/// Plans, in the order they rank.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, HwType)]
pub enum Plan {
    /// A step, by any amount: ranks below every other plan however large.
    Step(#[range(-9, 9)] i8),
    /// Nothing.
    Rest,
    /// Two fields, compared in order.
    Strike {
        /// How hard.
        #[range(0, 3)]
        power: u8,
        /// Signed, so negative values rank low.
        #[range(-2, 2)]
        bias: i8,
    },
    /// A struct inside a variant.
    Go(Spot),
}

/// Which plan, kind or reason ranks higher, each comparison as Rust derives it.
#[handwired]
pub fn rank(a: Plan, b: Plan) -> (bool, bool, bool, bool, Plan, Plan) {
    (a < b, a <= b, a > b, a >= b, a.min(b), a.max(b))
}

fn plans() -> Vec<Plan> {
    let mut plans = vec![Plan::Rest];
    for step in [-9, -1, 0, 1, 9] {
        plans.push(Plan::Step(step));
    }
    for power in 0..=3 {
        for bias in -2..=2 {
            plans.push(Plan::Strike { power, bias });
        }
    }
    for x in [-3, 0, 3] {
        for y in [0, 3] {
            plans.push(Plan::Go(Spot { x, y }));
        }
    }
    plans
}

#[test]
fn enums_with_data_order_by_variant_then_fields() {
    let compiled = rank_circuit();
    assert_eq!(compiled.verify(10_000).unwrap().skipped, 0);
    // Every pair: an earlier variant with larger fields ranks lower, the second field decides where the first ties,
    // and negative fields rank low.
    let plans = plans();
    for &a in &plans {
        for &b in &plans {
            assert_eq!(compiled.eval(&(a, b)).unwrap(), rank(a, b), "{a:?} vs {b:?}");
        }
    }
    assert!(rank(Plan::Step(9), Plan::Rest).0);
}

/// Options and Results, against each other and constants, with `clamp`.
#[handwired]
#[allow(clippy::type_complexity)]
pub fn bounds(
    #[range(-2, 2)] a: Option<i8>,
    #[range(-2, 2)] b: Option<i8>,
    #[range(0, 3)] r: Result<u8, Plan>,
    #[range(0, 3)] s: Result<u8, Plan>,
) -> (bool, bool, Option<i8>, bool, bool, Result<u8, Plan>, Option<i8>) {
    (a < b, a >= None, a.max(b), r < s, r > Ok(2u8), r.min(s), a.clamp(Some(-1i8), Some(1i8)))
}

#[test]
fn options_and_results_order_as_rust_derives() {
    let compiled = bounds_circuit();
    assert_eq!(compiled.verify(10_000).unwrap().skipped, 0);
    let options = [None, Some(-2), Some(0), Some(2)];
    let results = [Ok(0), Ok(3), Err(Plan::Step(-9)), Err(Plan::Rest), Err(Plan::Strike { power: 0, bias: 1 })];
    for a in options {
        for b in options {
            for r in results {
                for s in results {
                    assert_eq!(compiled.eval(&(a, b, r, s)).unwrap(), bounds(a, b, r, s), "{a:?} {b:?} {r:?} {s:?}");
                }
            }
        }
    }
    // None ranks below every Some, Ok below every Err.
    assert!(compiled.eval(&(None, Some(-2), Err(Plan::Step(-9)), Ok(3))).unwrap().0);
    assert!(compiled.eval(&(None, Some(-2), Ok(3), Err(Plan::Step(-9)))).unwrap().3);
}

/// Fieldless enums and bools take `min`, `max` and `clamp`; constants compare while the circuit is built.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, HwType)]
pub enum Tier {
    /// Low.
    Low,
    /// Mid.
    Mid,
    /// High.
    High,
}

/// Plans known while the circuit is built.
const FIXED: [Plan; 3] = [Plan::Strike { power: 1, bias: -1 }, Plan::Strike { power: 1, bias: 0 }, Plan::Step(9)];

/// `min`/`max`/`clamp` on fieldless enums and bools, and comparisons of constant enums with data.
#[handwired]
pub fn tiers(a: Tier, b: Tier, flag: bool) -> (Tier, Tier, bool, bool, bool) {
    let constant = FIXED[0] < FIXED[1];
    (a.max(b), a.clamp(Tier::Mid, Tier::High), flag.min(constant), Plan::Rest > FIXED[2], FIXED[2] >= FIXED[1])
}

#[test]
fn fieldless_enums_take_min_and_max() {
    let compiled = tiers_circuit();
    assert_eq!(compiled.verify(1_000).unwrap().skipped, 0);
    for a in [Tier::Low, Tier::Mid, Tier::High] {
        for b in [Tier::Low, Tier::Mid, Tier::High] {
            for flag in [false, true] {
                assert_eq!(compiled.eval(&(a, b, flag)).unwrap(), tiers(a, b, flag));
            }
        }
    }
}
