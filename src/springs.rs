//! Springs, ported from handwired's `emitSpring`, `emitApplySpringMatrix` and
//! `emitAnimatedObject`. The host writes each evaluation's step matrix into
//! source cells (see [`crate::timing`]), and the circuit applies it.

use crate::circuit::{Circuit, OpKind};
use crate::emit::{emit_relu, emit_select};
use crate::wire::{Lin, Term, Wire};

fn has_finite_range(circuit: &Circuit, wire: Wire) -> bool {
    let range = circuit.range(wire);
    (range.min as f32).is_finite() && (range.max as f32).is_finite()
}

fn emit_absolute(circuit: &mut Circuit, value: Wire) -> Wire {
    let positive = emit_relu(circuit, value);
    let negated = circuit.affine(-value);
    let negative = emit_relu(circuit, negated);
    circuit.affine(positive + negative)
}

fn emit_clamp_zero_one(circuit: &mut Circuit, value: Wire, unit: Wire) -> Wire {
    let positive = emit_relu(circuit, value);
    let over = circuit.affine(value - unit);
    let above_one = emit_relu(circuit, over);
    let result = circuit.affine(positive - above_one);
    circuit.assume_range(result, 0.0, 1.0);
    result
}

/// A spring's next state after four explicit semi-implicit Euler steps, with every constant a wire.
#[derive(Copy, Clone, Debug)]
pub struct SpringState {
    pub position: Wire,
    pub velocity: Wire,
    /// 0 once the spring is within both settle thresholds (scaled), 1 while it moves, and between them near the
    /// thresholds.
    pub active: Wire,
}

/// Four steps of `velocity += (-stiffness × (position - destination) - damping × velocity) × timestep;
/// position += velocity × timestep`, then a soft settle: `active` is
/// `clamp01(relu(|position - destination| - settle_distance) × distance_scale + relu(|velocity| - settle_speed) ×
/// speed_scale)`, the position moves to the destination as `active` goes to 0, and the velocity scales by it.
/// `unit` must hold 1. [`emit_animated_object`] is the exact, matrix-stepped form.
#[allow(clippy::too_many_arguments)]
pub fn emit_spring(
    circuit: &mut Circuit,
    position_source: Wire,
    destination: Wire,
    velocity_source: Wire,
    stiffness: Wire,
    damping: Wire,
    timestep: Wire,
    settle_distance: Wire,
    settle_speed: Wire,
    settle_distance_scale: Wire,
    settle_speed_scale: Wire,
    unit: Wire,
) -> SpringState {
    let (mut position, mut velocity) = (position_source, velocity_source);
    for _ in 0..4 {
        let displacement = circuit.affine(position - destination);
        let spring_force = circuit.multiply(stiffness, displacement);
        let damping_force = circuit.multiply(damping, velocity);
        let acceleration = circuit.affine(-Lin::from(spring_force) - damping_force);
        let velocity_change = circuit.multiply(acceleration, timestep);
        velocity = circuit.affine(velocity + velocity_change);
        let position_change = circuit.multiply(velocity, timestep);
        position = circuit.affine(position + position_change);
    }
    let offset = circuit.affine(position - destination);
    let distance = emit_absolute(circuit, offset);
    let distance_over = circuit.affine(distance - settle_distance);
    let distance_above = emit_relu(circuit, distance_over);
    let speed = emit_absolute(circuit, velocity);
    let speed_over = circuit.affine(speed - settle_speed);
    let speed_above = emit_relu(circuit, speed_over);
    let scaled_distance = circuit.multiply(distance_above, settle_distance_scale);
    let scaled_speed = circuit.multiply(speed_above, settle_speed_scale);
    let combined = circuit.affine(scaled_distance + scaled_speed);
    let active = emit_clamp_zero_one(circuit, combined, unit);
    SpringState {
        position: emit_select(circuit, active, destination, position),
        velocity: circuit.multiply(active, velocity),
        active,
    }
}

/// The 4 cells of a spring matrix that the host writes before each evaluation, e.g.
/// `spring_matrix_power(spring_step_matrix(constants, 4.0), steps)` from [`crate::timing`]:
/// `displacement' = m[0] × displacement + m[1] × velocity`, `velocity' = m[2] × displacement + m[3] × velocity`.
pub type SpringMatrixWires = [Wire; 4];

/// One spring's displacement from its destination and its velocity after every step the matrix describes, e.g.
/// all of one evaluation's 4 ms steps at once. The matrix cells need finite declared ranges, e.g.
/// ±[`spring_matrix_bound`](crate::timing::spring_matrix_bound), so a host that writes an overflowing matrix fails
/// the source range check. Each product goes through `multiply`, so the backward pass differentiates whole
/// products. Cost: 14 operations in 3 layers.
pub fn emit_apply_spring_matrix(
    circuit: &mut Circuit,
    matrix: SpringMatrixWires,
    displacement: Wire,
    velocity: Wire,
) -> (Wire, Wire) {
    for entry in matrix {
        assert!(
            has_finite_range(circuit, entry),
            "spring matrix wires need finite declared ranges, e.g. ±spring_matrix_bound(constants)"
        );
    }
    let products = [
        circuit.multiply(matrix[0], displacement),
        circuit.multiply(matrix[1], velocity),
        circuit.multiply(matrix[2], displacement),
        circuit.multiply(matrix[3], velocity),
    ];
    let next_displacement = circuit.round_affine(products[0] + products[1]);
    let next_velocity = circuit.round_affine(products[2] + products[3]);
    (next_displacement, next_velocity)
}

/// One spring of an [`emit_animated_object`].
#[derive(Copy, Clone, Debug)]
pub struct AnimatedSpring {
    pub matrix: SpringMatrixWires,
    /// The displacement from `destination_during_steps`, usually a state cell the previous evaluation wrote back.
    pub displacement: Wire,
    pub velocity: Wire,
    /// The destination from this evaluation's time on, after its events.
    pub destination: Wire,
    /// The destination the stored displacement is measured from, in effect during this evaluation's steps,
    /// usually the destination the previous evaluation wrote back. `None` when the evaluation can't change the
    /// destination; the steps then pull toward `destination`.
    pub destination_during_steps: Option<Wire>,
}

#[derive(Copy, Clone, Debug)]
pub struct AnimatedObjectOptions {
    /// An exact bit: 1 when this evaluation ends at a fixed simulated settle time
    /// ([`PlannedEvaluation::settle_check`](crate::timing::PlannedEvaluation::settle_check)). The object only
    /// settles then, so settling doesn't depend on the frame rate.
    pub settle_check: Wire,
    /// An exact bit that pins every spring onto its destination with zero velocity, e.g. while dragging, or a
    /// planned snap.
    pub snap: Option<Wire>,
    /// Settle thresholds, 0.01 px and 0.01 px/s by default; each at least 2^-22.
    pub settle_distance: f64,
    pub settle_speed: f64,
}

impl AnimatedObjectOptions {
    pub fn new(settle_check: Wire) -> AnimatedObjectOptions {
        AnimatedObjectOptions { settle_check, snap: None, settle_distance: 0.01, settle_speed: 0.01 }
    }
}

#[derive(Clone, Debug)]
pub struct AnimatedObject {
    /// Each spring's displacement from `destination` after this evaluation: write it back, with `destination` as
    /// the next evaluation's `destination_during_steps`.
    pub displacements: Vec<Wire>,
    pub velocities: Vec<Wire>,
    /// What to draw, `destination + displacement`, rounded once. Not state, so the rounding never builds up.
    pub positions: Vec<Wire>,
    /// 1 when this evaluation checks settling and every spring is within both thresholds, or when every spring is
    /// exactly at rest (every displacement and velocity ±0).
    pub settled: Wire,
    /// 1 while the object keeps moving: not settled and not snapped. A host stops its clock once every object's
    /// active bit is 0.
    pub active: Wire,
}

/// Springs of one animated object, e.g. a window's x, y, width, height and scale, stepped by their matrices and
/// settled together with one exact bit. Each spring stores its displacement from its destination rather than a
/// position, so FP32 rounds the state at the size of the displacement, finely near rest even far from the origin.
/// When the object settles or snaps, every displacement and velocity becomes 0, up to the sign of zero.
///
/// For each spring, `(displacement', velocity') = matrix × (displacement, velocity)`, then
/// `displacement'' = f(f(destination_during_steps - destination) + displacement')`, so an unchanged destination
/// adds exactly 0. The exceedance, the sum of `relu(|displacement''| - distance)` and `relu(|velocity'| - speed)`,
/// is 0 exactly when every spring is within both thresholds; `settled = boolean(relu(0.5 × settle_check + 0.5 ×
/// at_rest - 2^20 × exceedance))`.
pub fn emit_animated_object(
    circuit: &mut Circuit,
    springs: &[AnimatedSpring],
    options: AnimatedObjectOptions,
) -> AnimatedObject {
    assert!(!springs.is_empty(), "an animated object needs at least one spring");
    let settle_distance = options.settle_distance as f32 as f64;
    let settle_speed = options.settle_speed as f32 as f64;
    for threshold in [settle_distance, settle_speed] {
        assert!(
            threshold.is_finite() && threshold >= 2f64.powi(-22),
            "settle thresholds must be finite and at least 2^-22, got {threshold}"
        );
    }
    for bit in std::iter::once(options.settle_check).chain(options.snap) {
        let range = circuit.range(bit);
        assert!(range.min >= 0.0 && range.max <= 1.0, "settle_check and snap must be declared within 0..1");
    }
    let mut exceedance_terms: Vec<Term> = Vec::new();
    let mut rest_terms: Vec<Term> = Vec::new();
    let mut stepped = Vec::with_capacity(springs.len());
    for spring in springs {
        let (next_displacement, next_velocity) =
            emit_apply_spring_matrix(circuit, spring.matrix, spring.displacement, spring.velocity);
        // Its own operation, so each threshold ReLU reads one FP32 value.
        let displacement = match spring.destination_during_steps {
            Some(during) if during != spring.destination => circuit
                .round_affine(Lin::new(vec![(during, 1.0), (spring.destination, -1.0), (next_displacement, 1.0)], 0.0)),
            _ => next_displacement,
        };
        for (value, threshold) in [(displacement, settle_distance), (next_velocity, settle_speed)] {
            let positive_part = circuit.reglu(1.0, value);
            let negative_part = circuit.reglu(1.0, -value);
            // relu(|value| - threshold), with |value| = relu(value) + relu(-value).
            exceedance_terms.push((circuit.reglu(1.0, positive_part + negative_part - threshold), 1.0));
            rest_terms.push((positive_part, 2f64.powi(127)));
            rest_terms.push((negative_part, 2f64.powi(127)));
        }
        stepped.push((displacement, next_velocity));
    }
    let exceedance = circuit.round_affine(Lin::new(exceedance_terms, 0.0));
    // Each lifted term is 0, at least 2^127 × 2^-149 = 2^-22, or infinite, so relu(1 - 2^22 × lifted rest) is the
    // at-rest bit. Kept as written: combining a repeated part would give 2^128, which isn't an FP32 value.
    let lifted_rest = circuit.append(OpKind::Reglu, Lin::constant(1.0), Lin::new(rest_terms, 0.0), true, None);
    let at_rest = circuit.reglu(1.0, Lin::new(vec![(lifted_rest, -(2f64.powi(22)))], 1.0));
    // The settle check and rest bits first, so the partial sum is exactly 0, 0.5 or 1 before the exceedance.
    let score_terms = vec![(options.settle_check, 0.5), (at_rest, 0.5), (exceedance, -(2f64.powi(20)))];
    let settled_score = circuit.append(OpKind::Reglu, Lin::constant(1.0), Lin::new(score_terms, 0.0), true, None);
    circuit.mark_rounding_point(settled_score);
    let settled = circuit.boolean(settled_score);
    let not_settled = circuit.round_affine(1.0 - settled);
    let active = match options.snap {
        None => not_settled,
        Some(snap) => {
            let not_snapped = circuit.round_affine(1.0 - snap);
            circuit.multiply(not_settled, not_snapped)
        }
    };
    let displacements: Vec<Wire> =
        stepped.iter().map(|&(displacement, _)| circuit.multiply(displacement, active)).collect();
    let velocities = stepped.iter().map(|&(_, velocity)| circuit.multiply(velocity, active)).collect();
    let positions = springs
        .iter()
        .zip(&displacements)
        .map(|(spring, &displacement)| circuit.round_affine(spring.destination + displacement))
        .collect();
    AnimatedObject { displacements, velocities, positions, settled, active }
}
