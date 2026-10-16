//! Gather lowering: a sum of products of a declared one-hot group's members
//! with candidate values compiles to a read of the selected candidate.
//!
//! Ported from handwired's `planGatherLowering`. A matching sum adds f(0 × value)
//! = ±0 for every unselected member and the selected value once, so reading the
//! selected candidate and adding it to the bias gives the same FP32 result, up
//! to the sign of an exactly zero result. Candidates need finite FP32 bounds,
//! since an unselected f(Infinity × 0) would make the dense sum NaN.

use std::collections::HashMap;

use crate::circuit::{Circuit, Macro, OneHotGroup, OpKind, expression_key};
use crate::wire::{Term, Wire};

/// A lowered sum, keyed by its operation index: the candidate of each group position, in group order.
#[derive(Clone, Debug)]
pub(crate) struct GatherPlan {
    pub group: usize,
    pub candidates: Vec<Wire>,
}

struct MaskedProduct {
    coefficient: f64,
    left: Wire,
    gate: Wire,
    gate_coefficient: f64,
    exact_multiply: bool,
}

pub(crate) fn plan_gather_lowering(circuit: &Circuit) -> HashMap<usize, GatherPlan> {
    let mut plans = HashMap::new();
    if circuit.one_hot_groups.is_empty() {
        return plans;
    }
    let mut groups_by_wire: HashMap<Wire, Vec<usize>> = HashMap::new();
    let positions_by_group: Vec<HashMap<Wire, usize>> = circuit
        .one_hot_groups
        .iter()
        .enumerate()
        .map(|(group_index, group)| {
            let mut positions = HashMap::new();
            for (position, &member) in group.members.iter().enumerate() {
                positions.insert(member, position);
                groups_by_wire.entry(member).or_default().push(group_index);
            }
            positions
        })
        .collect();
    for (operation_index, operation) in circuit.operations.iter().enumerate() {
        if operation.kind != OpKind::Affine || operation.left_terms.is_empty() {
            continue;
        }
        let products: Option<Vec<MaskedProduct>> = operation
            .left_terms
            .iter()
            .map(|&(wire, coefficient)| {
                if coefficient != 1.0 && coefficient != -1.0 {
                    return None;
                }
                parse_masked_product(circuit, wire, coefficient)
            })
            .collect();
        let Some(products) = products else { continue };
        // Every term of a matching sum reaches the group through its left or gate wire, so the first product's
        // groups are the only ones to try.
        let first = &products[0];
        let mut candidate_groups: Vec<usize> = Vec::new();
        for wire in [first.gate, first.left] {
            for &group in groups_by_wire.get(&wire).into_iter().flatten() {
                if !candidate_groups.contains(&group) {
                    candidate_groups.push(group);
                }
            }
        }
        for group in candidate_groups {
            let group_positions = &positions_by_group[group];
            if let Some(candidates) =
                attempt_gather_match(circuit, &products, &circuit.one_hot_groups[group], group_positions)
            {
                plans.insert(operation_index, GatherPlan { group, candidates });
                break;
            }
        }
    }
    plans
}

/// Undo the builder's affine composition for one side of a ReGLU: a small affine wire is copied into its readers, so
/// a product against it stores its definition. The deduplication map recovers the wire. A sign of -1 means the side
/// is the negated wire, as in the negative arm of a mixed-sign product.
fn resolve_affine_side(circuit: &Circuit, terms: &[Term], bias: f64, allow_negated: bool) -> Option<(Wire, f64)> {
    if bias == 0.0 && terms.len() == 1 {
        let (wire, coefficient) = terms[0];
        if coefficient == 1.0 {
            return Some((wire, 1.0));
        }
        if coefficient == -1.0 && allow_negated {
            return Some((wire, -1.0));
        }
    }
    if terms.is_empty() {
        return None;
    }
    if let Some(&wire) = circuit.expression_wires.get(&expression_key(OpKind::Affine, terms, bias, &[], 0.0)) {
        return Some((wire, 1.0));
    }
    if !allow_negated {
        return None;
    }
    let negated: Vec<Term> = terms.iter().map(|&(wire, coefficient)| (wire, -coefficient)).collect();
    circuit.expression_wires.get(&expression_key(OpKind::Affine, &negated, -bias, &[], 0.0)).map(|&wire| (wire, -1.0))
}

/// One term of a candidate masked sum: a `multiply`, or a ReGLU whose left side is a wire and whose gate side is a
/// wire or its negation.
fn parse_masked_product(circuit: &Circuit, wire: Wire, coefficient: f64) -> Option<MaskedProduct> {
    let operation = circuit.operation(wire)?;
    if let Some(Macro::Multiply { left, right }) = operation.macro_op {
        return Some(MaskedProduct { coefficient, left, gate: right, gate_coefficient: 1.0, exact_multiply: true });
    }
    if operation.kind != OpKind::Reglu {
        return None;
    }
    let (left, _) = resolve_affine_side(circuit, &operation.left_terms, operation.left_bias, false)?;
    let (gate, gate_coefficient) = resolve_affine_side(circuit, &operation.gate_terms, operation.gate_bias, true)?;
    Some(MaskedProduct { coefficient, left, gate, gate_coefficient, exact_multiply: false })
}

/// Lowering needs finite FP32 bounds: a value that may be infinite turns an unselected product into NaN.
fn has_finite_range(circuit: &Circuit, wire: Wire) -> bool {
    let range = circuit.range(wire);
    (range.min as f32).is_finite() && (range.max as f32).is_finite()
}

/// The candidates by group position when the products are one complete masked read of the group, each term provably
/// equal to its masked read in FP32.
fn attempt_gather_match(
    circuit: &Circuit,
    products: &[MaskedProduct],
    group: &OneHotGroup,
    position_of: &HashMap<Wire, usize>,
) -> Option<Vec<Wire>> {
    // The negative arms of hand-written mixed-sign products: f(bit × max(0, value)) - f(bit × max(0, -value)) reads
    // a signed value only as a pair, so each arm must find its partner.
    let mut negative_arms: HashMap<(Wire, Wire), bool> = HashMap::new();
    for product in products {
        if product.coefficient == -1.0
            && product.gate_coefficient == -1.0
            && position_of.contains_key(&product.left)
            && !position_of.contains_key(&product.gate)
            && negative_arms.insert((product.left, product.gate), false).is_some()
        {
            return None;
        }
    }
    let mut options_by_term: Vec<Vec<(usize, Wire)>> = Vec::new();
    for product in products {
        let &MaskedProduct { coefficient, left, gate, gate_coefficient, exact_multiply } = product;
        if coefficient == 1.0 && gate_coefficient == 1.0 {
            let mut options = Vec::new();
            // The bit in the gate: f(value × max(0, bit)) selects any finite value.
            if let Some(&position) = position_of.get(&gate)
                && has_finite_range(circuit, left)
            {
                options.push((position, left));
            }
            if let Some(&position) = position_of.get(&left) {
                let gate_range = circuit.range(gate);
                if exact_multiply && has_finite_range(circuit, gate) {
                    options.push((position, gate));
                } else if let Some(used) = negative_arms.get_mut(&(left, gate))
                    && !*used
                    && has_finite_range(circuit, gate)
                {
                    *used = true;
                    options.push((position, gate));
                } else if gate_range.min >= 0.0 && gate_range.max.is_finite() {
                    // The bit on the left: f(bit × max(0, value)) reads the value only when the ReLU is the identity.
                    options.push((position, gate));
                }
            }
            if options.is_empty() {
                return None;
            }
            options_by_term.push(options);
        } else if gate_coefficient == -1.0 && position_of.contains_key(&gate) {
            // max(0, -bit) is 0 for a 0/1 bit, so the term adds a zero, unless the value may be infinite.
            if !has_finite_range(circuit, left) {
                return None;
            }
        } else if coefficient == -1.0 && gate_coefficient == -1.0 && position_of.contains_key(&left) {
            // The negative arm of a mixed-sign pair, checked as used below.
        } else {
            return None;
        }
    }
    if negative_arms.values().any(|&used| !used) {
        return None;
    }
    // Terms with one possible position first, then terms whose bit could be on either side take the open one.
    let mut candidates: Vec<Option<Wire>> = vec![None; group.members.len()];
    let mut flexible = Vec::new();
    for options in options_by_term {
        if let [(position, value)] = options[..] {
            if candidates[position].is_some() {
                return None;
            }
            candidates[position] = Some(value);
        } else {
            flexible.push(options);
        }
    }
    for options in flexible {
        let &(position, value) = options.iter().find(|&&(position, _)| candidates[position].is_none())?;
        candidates[position] = Some(value);
    }
    candidates.into_iter().collect()
}
