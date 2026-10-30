//! Springs and host timing, ported from handwired's tests/runtime/spring-matrix.test.ts.

mod common;

use std::collections::HashMap;

use common::Rng;
use handwired_rs::timing::{
    EvaluationKind, IDENTITY_SPRING_MATRIX, PlannedEvaluation, STOPPED_SPRING_CLOCK, SpringClock, SpringClockConfig,
    SpringConstants, WriteBack, advance_frame, count_spring_steps, multiply_spring_matrices, quantize_frame_timing,
    quantize_time_ms, spring_matrix_bound, spring_matrix_power, spring_step_matrix,
};
use handwired_rs::{
    AnimatedObject, AnimatedObjectOptions, AnimatedSpring, BackpropOptions, BooleanGradient, Circuit, CompileOptions,
    TensorExecutor, TensorProgram, Wire, emit_animated_object, emit_apply_spring_matrix,
};

const CONSTANTS: [SpringConstants; 3] = [
    SpringConstants { stiffness: 290.0, damping: 30.0 },
    SpringConstants { stiffness: 650.0, damping: 45.0 },
    SpringConstants { stiffness: 120.0, damping: 14.0 },
];

fn config() -> SpringClockConfig {
    SpringClockConfig::default()
}

fn matrix_bound() -> f64 {
    CONSTANTS.iter().map(|&constants| spring_matrix_bound(constants, &config())).fold(0.0, f64::max)
}

fn matrix_f32(constants: SpringConstants, steps: u32) -> [f32; 4] {
    spring_matrix_power(spring_step_matrix(constants, 4.0), steps).map(|entry| entry as f32)
}

fn counts(program: &TensorProgram) -> (usize, usize, usize) {
    (program.live_operation_count(), program.layers.len(), program.stored_weight_count())
}

fn next_up(value: f32) -> f32 {
    f32::from_bits(value.to_bits() + 1)
}

const SMALLEST_SUBNORMAL: f32 = 1e-45;

/// Every compile mode returns the reference evaluator's bits on every live wire.
fn check_every_compile_mode(circuit: &Circuit, roots: &[Wire], sources: &[Vec<f32>]) {
    let modes = [
        CompileOptions::default(),
        CompileOptions { lower_gathers: false, recompute_depths: false },
        CompileOptions { lower_gathers: true, recompute_depths: true },
    ];
    for options in modes {
        let program = circuit.compile_with(Some(roots), options);
        for source in sources {
            let reference = circuit.evaluate_reference(source).unwrap();
            let tensor = circuit.evaluate_tensor(source, &program).unwrap();
            for wire in &program.wire_order {
                let (got, want) = (tensor[wire.index()], reference[wire.index()]);
                assert!(got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()), "wire {}", wire.index());
            }
        }
    }
}

/// Float64 semi-implicit Euler steps.
fn euler_steps(constants: SpringConstants, displacement: f64, velocity: f64, steps: u32) -> (f64, f64) {
    let (mut e, mut v) = (displacement, velocity);
    for _ in 0..steps {
        v += (-constants.stiffness * e - constants.damping * v) * 0.004;
        e += v * 0.004;
    }
    (e, v)
}

/// emit_apply_spring_matrix written independently in FP32: each sum adds from +0 in stored order.
fn apply_matrix_fp32(matrix: [f32; 4], displacement: f32, velocity: f32) -> (f32, f32) {
    ((0.0 + matrix[0] * displacement) + matrix[1] * velocity, (0.0 + matrix[2] * displacement) + matrix[3] * velocity)
}

fn sum_fp32(bias: f64, terms: &[(f64, f32)]) -> f32 {
    let mut value = bias as f32;
    for &(coefficient, operand) in terms {
        value += coefficient as f32 * operand;
    }
    value
}

fn reglu_fp32(left: f32, gate: f32) -> f32 {
    let relu = if gate > 0.0 {
        gate
    } else if gate.is_nan() {
        f32::NAN
    } else {
        0.0
    };
    left * relu
}

fn boolean_fp32(score: f32) -> f32 {
    (score + 8_388_609.0) - 8_388_609.0
}

#[derive(Clone)]
struct SpringSample {
    matrix: [f32; 4],
    displacement: f32,
    velocity: f32,
    destination: f32,
    destination_during_steps: f32,
}

struct Twin {
    displacements: Vec<f32>,
    positions: Vec<f32>,
    velocities: Vec<f32>,
    settled: f32,
    active: f32,
    at_rest: f32,
}

/// emit_animated_object written independently in FP32.
fn animated_object_fp32(springs: &[SpringSample], separate: bool, settle_check: f32, snap: Option<f32>) -> Twin {
    let threshold = 0.01f32;
    let (mut exceedance, mut lifted_rest) = (0.0f32, 0.0f32);
    let lift = 2f32.powi(127);
    let stepped: Vec<(f32, f32)> = springs
        .iter()
        .map(|spring| {
            let (displacement, velocity) = apply_matrix_fp32(spring.matrix, spring.displacement, spring.velocity);
            let displacement = if separate {
                sum_fp32(
                    0.0,
                    &[(1.0, spring.destination_during_steps), (-1.0, spring.destination), (1.0, displacement)],
                )
            } else {
                displacement
            };
            for value in [displacement, velocity] {
                let positive = reglu_fp32(1.0, sum_fp32(0.0, &[(1.0, value)]));
                let negative = reglu_fp32(1.0, sum_fp32(0.0, &[(-1.0, value)]));
                exceedance += reglu_fp32(1.0, sum_fp32(-(threshold as f64), &[(1.0, positive), (1.0, negative)]));
                lifted_rest = (lifted_rest + lift * positive) + lift * negative;
            }
            (displacement, velocity)
        })
        .collect();
    let at_rest = reglu_fp32(1.0, sum_fp32(1.0, &[(-(2f64.powi(22)), reglu_fp32(1.0, lifted_rest))]));
    let score = reglu_fp32(1.0, sum_fp32(0.0, &[(0.5, settle_check), (0.5, at_rest), (-(2f64.powi(20)), exceedance)]));
    let settled = boolean_fp32(score);
    let not_settled = sum_fp32(1.0, &[(-1.0, settled)]);
    let active = match snap {
        None => not_settled,
        Some(snap) => {
            reglu_fp32(sum_fp32(0.0, &[(1.0, not_settled)]), sum_fp32(0.0, &[(1.0, sum_fp32(1.0, &[(-1.0, snap)]))]))
        }
    };
    let displacements: Vec<f32> =
        stepped.iter().map(|&(d, _)| reglu_fp32(sum_fp32(0.0, &[(1.0, d)]), sum_fp32(0.0, &[(1.0, active)]))).collect();
    Twin {
        positions: springs
            .iter()
            .zip(&displacements)
            .map(|(s, &d)| sum_fp32(0.0, &[(1.0, s.destination), (1.0, d)]))
            .collect(),
        velocities: stepped
            .iter()
            .map(|&(_, v)| reglu_fp32(sum_fp32(0.0, &[(1.0, v)]), sum_fp32(0.0, &[(1.0, active)])))
            .collect(),
        displacements,
        settled,
        active,
        at_rest,
    }
}

// ---- Applying a spring matrix ------------------------------------------------------------------------------------

fn matrix_circuit() -> (Circuit, Vec<Wire>) {
    let mut circuit = Circuit::new(6);
    for cell in 0..4 {
        circuit.assume_range(circuit.source(cell), -matrix_bound(), matrix_bound());
    }
    let matrix = [circuit.source(0), circuit.source(1), circuit.source(2), circuit.source(3)];
    let (displacement, velocity) = (circuit.source(4), circuit.source(5));
    let (next_displacement, next_velocity) = emit_apply_spring_matrix(&mut circuit, matrix, displacement, velocity);
    (circuit, vec![next_displacement, next_velocity])
}

#[test]
fn a_spring_matrix_equals_euler_steps_in_fp32_and_its_cost_is_pinned() {
    let (circuit, roots) = matrix_circuit();
    let program = circuit.compile(Some(&roots));
    assert_eq!(counts(&program), (14, 3, 28));
    let mut executor = TensorExecutor::new(&program);
    let mut rng = Rng::new(3);
    let (mut worst_displacement, mut worst_velocity) = (0f64, 0f64);
    let mut sources = Vec::new();
    for sample in 0..1500 {
        let constants = CONSTANTS[rng.below(3)];
        let steps = if rng.below(5) == 0 { rng.below(4) } else { rng.below(301) } as u32;
        let displacement = rng.uniform(-2000.0, 2000.0) as f32;
        let velocity = rng.uniform(-5000.0, 5000.0) as f32;
        let matrix = matrix_f32(constants, steps);
        let source = [matrix[0], matrix[1], matrix[2], matrix[3], displacement, velocity];
        let outputs = executor.run(&source).to_vec();
        let expected = apply_matrix_fp32(matrix, displacement, velocity);
        assert_eq!((outputs[0], outputs[1]), expected);
        let (e, v) = euler_steps(constants, displacement as f64, velocity as f64, steps);
        worst_displacement = worst_displacement.max((outputs[0] as f64 - e).abs());
        worst_velocity = worst_velocity.max((outputs[1] as f64 - v).abs());
        if sample < 40 {
            sources.push(source.to_vec());
        }
    }
    assert!(worst_displacement < 1e-3 && worst_velocity < 1e-2, "{worst_displacement} {worst_velocity}");
    check_every_compile_mode(&circuit, &roots, &sources);
}

#[test]
fn zero_steps_keep_the_state_and_extreme_states_stay_fp32_products() {
    let (circuit, roots) = matrix_circuit();
    let program = circuit.compile(Some(&roots));
    let mut executor = TensorExecutor::new(&program);
    for (displacement, velocity) in
        [(123.456f32, -7.5f32), (-0.0, 0.0), (1e-45, -1e-45), (f32::MAX, -f32::MAX), (16_777_218.0, 3.0)]
    {
        let identity = executor.run(&[1.0, 0.0, 0.0, 1.0, displacement, velocity]).to_vec();
        assert_eq!((identity[0] + 0.0, identity[1] + 0.0), (displacement + 0.0, velocity + 0.0));
        let matrix = matrix_f32(CONSTANTS[1], 300);
        let stepped = executor.run(&[matrix[0], matrix[1], matrix[2], matrix[3], displacement, velocity]).to_vec();
        assert_eq!((stepped[0], stepped[1]), apply_matrix_fp32(matrix, displacement, velocity));
    }
}

#[test]
#[should_panic(expected = "finite declared ranges")]
fn matrix_wires_need_finite_declared_ranges() {
    let mut circuit = Circuit::new(6);
    for cell in 0..3 {
        circuit.assume_range(circuit.source(cell), -10.0, 10.0);
    }
    let matrix = [circuit.source(0), circuit.source(1), circuit.source(2), circuit.source(3)];
    let (displacement, velocity) = (circuit.source(4), circuit.source(5));
    emit_apply_spring_matrix(&mut circuit, matrix, displacement, velocity);
}

// ---- Animated objects --------------------------------------------------------------------------------------------

/// Cells per spring: 4 matrix cells, displacement, velocity, destination, destination during steps. Then the
/// settle check and snap bits.
const FIELDS: usize = 8;

struct Object {
    circuit: Circuit,
    object: AnimatedObject,
    roots: Vec<Wire>,
    settle_check_cell: usize,
    snap_cell: usize,
}

fn build_object(springs: usize, separate: bool, with_snap: bool, thresholds: Option<(f64, f64)>) -> Object {
    let settle_check_cell = springs * FIELDS;
    let snap_cell = settle_check_cell + 1;
    let mut circuit = Circuit::new(snap_cell + 1);
    circuit.assume_range(circuit.source(settle_check_cell), 0.0, 1.0);
    circuit.assume_range(circuit.source(snap_cell), 0.0, 1.0);
    let bound = matrix_bound();
    let list: Vec<AnimatedSpring> = (0..springs)
        .map(|index| {
            let start = index * FIELDS;
            for cell in start..start + 4 {
                circuit.assume_range(circuit.source(cell), -bound, bound);
            }
            AnimatedSpring {
                matrix: [
                    circuit.source(start),
                    circuit.source(start + 1),
                    circuit.source(start + 2),
                    circuit.source(start + 3),
                ],
                displacement: circuit.source(start + 4),
                velocity: circuit.source(start + 5),
                destination: circuit.source(start + 6),
                destination_during_steps: separate.then(|| circuit.source(start + 7)),
            }
        })
        .collect();
    let mut options = AnimatedObjectOptions::new(circuit.source(settle_check_cell));
    if with_snap {
        options.snap = Some(circuit.source(snap_cell));
    }
    if let Some((distance, speed)) = thresholds {
        options.settle_distance = distance;
        options.settle_speed = speed;
    }
    let object = emit_animated_object(&mut circuit, &list, options);
    let mut roots = object.displacements.clone();
    roots.extend(&object.positions);
    roots.extend(&object.velocities);
    roots.extend([object.settled, object.active]);
    Object { circuit, object, roots, settle_check_cell, snap_cell }
}

fn write_springs(source: &mut [f32], springs: &[SpringSample]) {
    for (index, spring) in springs.iter().enumerate() {
        let start = index * FIELDS;
        source[start..start + 4].copy_from_slice(&spring.matrix);
        source[start + 4] = spring.displacement;
        source[start + 5] = spring.velocity;
        source[start + 6] = spring.destination;
        source[start + 7] = spring.destination_during_steps;
    }
}

#[test]
fn animated_object_costs_match_handwireds() {
    let cost = |springs, separate, snap| {
        let object = build_object(springs, separate, snap, None);
        counts(&object.circuit.compile(Some(&object.roots)))
    };
    assert_eq!(cost(1, false, false), (29, 11, 54));
    assert_eq!(cost(1, true, false), (30, 12, 57));
    assert_eq!(cost(2, false, false), (52, 11, 102));
    assert_eq!(cost(5, true, true), (128, 13, 264));
}

#[test]
fn animated_objects_match_an_independent_fp32_twin_bit_for_bit() {
    let threshold = 0.01f32;
    let above = next_up(threshold);
    let mut rng = Rng::new(17);
    let (mut settled_samples, mut not_settled_near, mut settled_at_rest_without_check, mut subnormal_not_at_rest) =
        (0, 0, 0, 0);
    for separate in [false, true] {
        for with_snap in [false, true] {
            let object = build_object(3, separate, with_snap, None);
            let program = object.circuit.compile(Some(&object.roots));
            let mut executor = TensorExecutor::new(&program);
            let mut sources = Vec::new();
            for sample in 0..1500 {
                let mut source = vec![0.0f32; object.circuit.source_size()];
                let pick = rng.uniform(0.0, 1.0);
                let near_rest = pick < 0.45;
                let at_rest = (0.45..0.6).contains(&pick);
                let subnormal_spring = if at_rest && rng.below(4) == 0 { rng.below(3) as i64 } else { -1 };
                let inside = rng.below(2) == 0;
                let near = |rng: &mut Rng, inside: bool| -> f32 {
                    let pick = rng.uniform(0.0, 1.0);
                    let sign: f32 = if rng.below(2) == 0 { -1.0 } else { 1.0 };
                    if inside {
                        if pick < 0.4 { sign * threshold } else { sign * rng.uniform(0.0, 0.01) as f32 }
                    } else if pick < 0.3 {
                        sign * threshold
                    } else if pick < 0.5 {
                        sign * above
                    } else {
                        rng.uniform(-0.02, 0.02) as f32
                    }
                };
                let springs: Vec<SpringSample> = (0..3)
                    .map(|index| {
                        let constants = CONSTANTS[rng.below(3)];
                        let steps = if near_rest || at_rest { 0 } else { rng.below(301) as u32 };
                        let destination = if rng.below(5) < 2 {
                            0.0
                        } else {
                            (rng.below(801) as f64 + if rng.below(2) == 0 { 0.0 } else { rng.uniform(0.0, 1.0) }) as f32
                        };
                        let destination_during_steps = if at_rest || rng.below(2) == 0 {
                            destination
                        } else {
                            (destination as f64 + rng.uniform(-50.0, 50.0)) as f32
                        };
                        let matrix = matrix_f32(constants, steps);
                        if at_rest {
                            let subnormal = index as i64 == subnormal_spring;
                            let on_velocity = rng.below(2) == 0;
                            let zero = |rng: &mut Rng| if rng.below(2) == 0 { -0.0 } else { 0.0 };
                            return SpringSample {
                                matrix,
                                displacement: if subnormal && !on_velocity {
                                    -SMALLEST_SUBNORMAL
                                } else {
                                    zero(&mut rng)
                                },
                                velocity: if subnormal && on_velocity { SMALLEST_SUBNORMAL } else { zero(&mut rng) },
                                destination,
                                destination_during_steps,
                            };
                        }
                        SpringSample {
                            matrix,
                            displacement: if near_rest {
                                near(&mut rng, inside)
                            } else {
                                rng.uniform(-900.0, 900.0) as f32
                            },
                            velocity: if near_rest {
                                near(&mut rng, inside)
                            } else {
                                rng.uniform(-3000.0, 3000.0) as f32
                            },
                            destination,
                            destination_during_steps,
                        }
                    })
                    .collect();
                write_springs(&mut source, &springs);
                source[object.settle_check_cell] = if rng.below(10) < 7 { 1.0 } else { 0.0 };
                source[object.snap_cell] = if with_snap && rng.below(100) < 15 { 1.0 } else { 0.0 };
                let outputs = executor.run(&source).to_vec();
                let twin = animated_object_fp32(
                    &springs,
                    separate,
                    source[object.settle_check_cell],
                    with_snap.then_some(source[object.snap_cell]),
                );
                let mut expected = twin.displacements.clone();
                expected.extend(&twin.positions);
                expected.extend(&twin.velocities);
                expected.extend([twin.settled, twin.active]);
                let bits = |values: &[f32]| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&outputs), bits(&expected), "sample {sample}");
                if twin.settled == 1.0 {
                    settled_samples += 1;
                    for (index, spring) in springs.iter().enumerate() {
                        assert!(outputs[index] == 0.0 && outputs[6 + index] == 0.0);
                        assert_eq!(outputs[3 + index], spring.destination);
                    }
                    if at_rest && source[object.settle_check_cell] == 0.0 {
                        settled_at_rest_without_check += 1;
                    }
                } else if near_rest && source[object.settle_check_cell] == 1.0 {
                    not_settled_near += 1;
                }
                if subnormal_spring >= 0 {
                    assert_eq!(twin.at_rest, 0.0);
                    if source[object.settle_check_cell] == 0.0 {
                        assert_eq!(outputs[9], 0.0);
                        subnormal_not_at_rest += 1;
                    }
                }
                if sample < 30 {
                    sources.push(source);
                }
            }
            check_every_compile_mode(&object.circuit, &object.roots, &sources);
        }
    }
    assert!(settled_samples > 500 && not_settled_near > 300, "{settled_samples} {not_settled_near}");
    assert!(settled_at_rest_without_check > 200 && subnormal_not_at_rest > 30);
}

#[test]
fn objects_settle_at_a_settle_check_or_at_exact_rest_and_huge_states_never_flip_the_bit() {
    let object = build_object(1, false, false, None);
    let program = object.circuit.compile(Some(&object.roots));
    let mut executor = TensorExecutor::new(&program);
    let threshold = 0.01f32;
    // [displacement, position, velocity, settled, active], ignoring the sign of zero.
    let mut run = |displacement: f32, velocity: f32, settle_check: f32, destination: f32| -> Vec<f32> {
        let mut source = vec![0.0f32; object.circuit.source_size()];
        write_springs(
            &mut source,
            &[SpringSample {
                matrix: [1.0, 0.0, 0.0, 1.0],
                displacement,
                velocity,
                destination,
                destination_during_steps: destination,
            }],
        );
        source[object.settle_check_cell] = settle_check;
        executor.run(&source).iter().map(|value| value + 0.0).collect()
    };
    assert_eq!(run(threshold, -threshold, 1.0, 0.0), [0.0, 0.0, 0.0, 1.0, 0.0]);
    assert_eq!(run(-threshold, threshold, 1.0, 0.0), [0.0, 0.0, 0.0, 1.0, 0.0]);
    let above = next_up(threshold);
    assert_eq!(run(above, 0.0, 1.0, 0.0), [above, above, 0.0, 0.0, 1.0]);
    assert_eq!(run(0.0, -above, 1.0, 0.0), [0.0, 0.0, -above, 0.0, 1.0]);
    assert_eq!(run(threshold, 0.0, 0.0, 0.0), [threshold, threshold, 0.0, 0.0, 1.0]);
    assert_eq!(run(0.0, 0.0, 0.0, 0.0), [0.0, 0.0, 0.0, 1.0, 0.0]);
    assert_eq!(run(-0.0, -0.0, 0.0, 0.0), [0.0, 0.0, 0.0, 1.0, 0.0]);
    assert_eq!(run(SMALLEST_SUBNORMAL, 0.0, 0.0, 0.0), [SMALLEST_SUBNORMAL, SMALLEST_SUBNORMAL, 0.0, 0.0, 1.0]);
    assert_eq!(run(0.0, -SMALLEST_SUBNORMAL, 0.0, 0.0), [0.0, 0.0, -SMALLEST_SUBNORMAL, 0.0, 1.0]);
    assert_eq!(run(f32::MAX, f32::MAX, 1.0, 0.0)[3..], [0.0, 1.0]);
    assert_eq!(run(f32::MAX, f32::MAX, 0.0, 0.0)[3..], [0.0, 1.0]);
    // At 100,000 px, positions are 1/128 px apart, and the stored displacement still settles at exactly 0.01.
    assert_eq!(run(threshold, 0.0, 1.0, 100_000.0), [0.0, 100_000.0, 0.0, 1.0, 0.0]);
    assert_eq!(run(above, 0.0, 1.0, 100_000.0), [above, 100_000.0 + above, 0.0, 0.0, 1.0]);
}

#[test]
fn the_smallest_threshold_still_resolves_and_bad_thresholds_are_refused() {
    let floor = 2f64.powi(-22);
    let object = build_object(1, false, false, Some((floor, floor)));
    let program = object.circuit.compile(Some(&object.roots));
    let mut source = vec![0.0f32; object.circuit.source_size()];
    write_springs(
        &mut source,
        &[SpringSample {
            matrix: [1.0, 0.0, 0.0, 1.0],
            displacement: next_up(floor as f32),
            velocity: 0.0,
            destination: 0.0,
            destination_during_steps: 0.0,
        }],
    );
    source[object.settle_check_cell] = 1.0;
    assert_eq!(TensorExecutor::new(&program).run(&source)[3], 0.0);
    source[4] = floor as f32;
    assert_eq!(TensorExecutor::new(&program).run(&source)[3], 1.0);
    let kinds: Vec<_> = object.circuit.rounding_points().into_iter().map(|(_, kind)| kind).collect();
    assert_eq!(kinds.iter().filter(|&&kind| kind == handwired_rs::RoundingPointKind::Marked).count(), 1);
    assert_eq!(kinds.iter().filter(|&&kind| kind == handwired_rs::RoundingPointKind::Boolean).count(), 1);
    for thresholds in [(2f64.powi(-23), 0.01), (0.01, f64::NAN)] {
        let refused = std::panic::catch_unwind(|| build_object(1, false, false, Some(thresholds)));
        assert!(refused.is_err());
    }
}

#[test]
fn a_changed_destination_rebases_once_and_an_unchanged_one_adds_exactly_zero() {
    let object = build_object(1, true, false, None);
    let program = object.circuit.compile(Some(&object.roots));
    let mut executor = TensorExecutor::new(&program);
    let mut rng = Rng::new(23);
    let mut rebased = 0;
    for _ in 0..2000 {
        let matrix = matrix_f32(CONSTANTS[0], rng.below(301) as u32);
        let displacement = rng.uniform(-900.0, 900.0) as f32;
        let velocity = rng.uniform(-3000.0, 3000.0) as f32;
        let during = (1_048_576 + rng.below(801)) as f32;
        let destination = if rng.below(2) == 0 { during } else { (1_048_576 + rng.below(801)) as f32 };
        let mut source = vec![0.0f32; object.circuit.source_size()];
        write_springs(
            &mut source,
            &[SpringSample { matrix, displacement, velocity, destination, destination_during_steps: during }],
        );
        let outputs = executor.run(&source).to_vec();
        let (stepped, _) = apply_matrix_fp32(matrix, displacement, velocity);
        if destination == during {
            assert_eq!(outputs[0] + 0.0, stepped + 0.0);
        } else {
            assert_eq!(outputs[0], during - destination + stepped);
            rebased += 1;
        }
    }
    assert!(rebased > 800);
}

#[test]
fn inlining_keeps_the_values_and_the_rounding_points() {
    let object = build_object(2, true, true, None);
    let mut target = Circuit::new(object.circuit.source_size());
    let cells: Vec<Wire> = (0..object.circuit.source_size()).map(Wire::from_index).collect();
    let map = target.inline_circuit(&object.circuit, &cells, false);
    let mut rng = Rng::new(5);
    for _ in 0..200 {
        let mut source = vec![0.0f32; object.circuit.source_size()];
        let springs: Vec<SpringSample> = (0..2)
            .map(|_| SpringSample {
                matrix: matrix_f32(CONSTANTS[0], rng.below(40) as u32),
                displacement: if rng.below(5) == 0 { 0.0 } else { rng.uniform(0.0, 0.03) as f32 },
                velocity: if rng.below(5) == 0 { 0.0 } else { rng.uniform(-0.02, 0.02) as f32 },
                destination: 0.0,
                destination_during_steps: if rng.below(2) == 0 { 0.0 } else { rng.uniform(0.0, 0.01) as f32 },
            })
            .collect();
        write_springs(&mut source, &springs);
        source[object.settle_check_cell] = if rng.below(2) == 0 { 1.0 } else { 0.0 };
        let expected = object.circuit.evaluate_reference(&source).unwrap();
        let actual = target.evaluate_reference(&source).unwrap();
        for root in &object.roots {
            assert_eq!(actual[map[root.index()].index()].to_bits(), expected[root.index()].to_bits());
        }
    }
    let kinds = |circuit: &Circuit| {
        let mut kinds: Vec<String> = circuit.rounding_points().iter().map(|(_, kind)| format!("{kind:?}")).collect();
        kinds.sort();
        kinds
    };
    assert_eq!(kinds(&target), kinds(&object.circuit));
}

#[test]
fn the_backward_pass_gives_matrix_cells_and_state_the_gradients_of_whole_products() {
    let object = build_object(1, false, false, None);
    let circuit = &object.circuit;
    let mut rng = Rng::new(29);
    for sample in 0..100 {
        let steps = if sample < 10 { 0 } else { rng.below(301) as u32 };
        let matrix = matrix_f32(CONSTANTS[sample % 3], steps);
        let destination = rng.uniform(0.0, 800.0) as f32;
        let displacement = rng.uniform(-500.0, 500.0) as f32;
        let velocity = rng.uniform(-2000.0, 2000.0) as f32;
        let mut source = vec![0.0f32; circuit.source_size()];
        write_springs(
            &mut source,
            &[SpringSample { matrix, displacement, velocity, destination, destination_during_steps: destination }],
        );
        let values = circuit.evaluate_reference(&source).unwrap();
        let mut seed = vec![0.0f32; values.len()];
        seed[object.object.positions[0].index()] = 1.0;
        assert!(circuit.backpropagate_reference(&values, &seed, BackpropOptions::default()).is_err());
        let options = BackpropOptions { boolean_gradient: BooleanGradient::StraightThrough, ..Default::default() };
        let gradients = circuit.backpropagate_reference(&values, &seed, options).unwrap();
        let close = |actual: f32, expected: f32| {
            assert!((actual - expected).abs() <= 1e-4 * expected.abs().max(1.0), "{actual} vs {expected}");
        };
        // position = destination + m0 × displacement + m1 × velocity
        close(gradients[0], displacement);
        close(gradients[1], velocity);
        assert_eq!((gradients[2], gradients[3]), (0.0, 0.0));
        close(gradients[4], matrix[0]);
        close(gradients[5], matrix[1]);
        assert_eq!(gradients[6], 1.0);
    }
}

// ---- Host timing ------------------------------------------------------------------------------------------------

#[test]
fn frame_durations_add_up_and_late_and_early_times_clamp() {
    let c = config();
    assert_eq!(quantize_time_ms(16.6667, &c), 16.671875);
    assert!(quantize_time_ms(-0.001, &c).is_sign_positive());
    let mut rng = Rng::new(41);
    for _ in 0..50 {
        let (mut previous, mut total, mut first, mut last) = (None, 0.0, 0.0, 0.0);
        let mut now = rng.uniform(0.0, 1e7);
        for _ in 0..60 {
            now += rng.uniform(0.0, 60.0);
            let events: Vec<f64> = (0..3).map(|_| now - rng.uniform(0.0, 50.0)).collect();
            let timing = quantize_frame_timing(now, previous, &events, &c);
            for &value in std::iter::once(&timing.duration_ms).chain(&timing.event_times_ms) {
                assert_eq!((value * 64.0).fract(), 0.0);
                assert_eq!(value as f32 as f64, value);
                assert!(!(value == 0.0 && value.is_sign_negative()));
            }
            for (index, &event_time) in timing.event_times_ms.iter().enumerate() {
                let quantized = quantize_time_ms(events[index], &c);
                match previous {
                    Some(previous) if quantized < previous => assert_eq!(timing.frame_time_ms + event_time, previous),
                    _ => assert_eq!(timing.frame_time_ms + event_time, quantized),
                }
            }
            total += timing.duration_ms;
            if previous.is_none() {
                first = timing.frame_time_ms;
            }
            previous = Some(timing.frame_time_ms);
            last = timing.frame_time_ms;
        }
        assert_eq!(total, last - first);
    }
    let timing = quantize_frame_timing(5000.0, None, &[4990.004, 4000.0], &c);
    assert_eq!((timing.frame_time_ms, timing.duration_ms, timing.event_times_ms), (5000.0, 0.0, vec![-10.0, -1000.0]));
    assert_eq!(quantize_frame_timing(5000.0, None, &[4990.01], &c).event_times_ms, [-9.984375]);
    let early = quantize_frame_timing(990.0, Some(1000.0), &[], &c);
    assert_eq!((early.frame_time_ms, early.duration_ms), (1000.0, 0.0));
    let stalled = quantize_frame_timing(101_000.0, Some(1000.0), &[101_003.0, 500.0], &c);
    assert_eq!(stalled.duration_ms, 1200.0);
    assert!(stalled.event_times_ms[0] == 0.0 && stalled.event_times_ms[0].is_sign_positive());
    assert_eq!(stalled.event_times_ms[1], -1200.0);
    assert_eq!(
        quantize_frame_timing(200_000.0, Some(1000.0), &[500.0, 1500.0, 199_000.0], &c).event_times_ms,
        [-1200.0, -1200.0, -1000.0]
    );
    assert_eq!(quantize_frame_timing(5000.0, None, &[1000.0], &c).event_times_ms, [-1200.0]);
    assert_eq!(quantize_frame_timing(1100.0, Some(1000.0), &[900.0], &c).event_times_ms, [-100.0]);
}

#[test]
fn step_counts_carry_the_leftover_or_snap_past_the_budget() {
    let c = config();
    let (mut checked, mut snaps) = (0, 0);
    for steps in 0..=305u32 {
        for offset in [-1.0 / 64.0, 0.0, 1.0 / 64.0] {
            let pending = 4.0 * steps as f64 + offset;
            if pending < 0.0 {
                continue;
            }
            for budget in [300, 7, 0] {
                let count = count_spring_steps(pending, budget, &c);
                let pending_steps = (pending / 4.0).floor();
                if pending_steps > budget as f64 {
                    assert!(count.snap && count.steps == 0 && count.leftover_ms == 0.0);
                    snaps += 1;
                } else {
                    assert_eq!(
                        (count.steps as f64, count.leftover_ms, count.snap),
                        (pending_steps, pending - 4.0 * pending_steps, false)
                    );
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 2700 && snaps > 1500);
    let full = count_spring_steps(1203.984375, 300, &c);
    assert_eq!((full.steps, full.leftover_ms, full.snap), (300, 3.984375, false));
    assert!(count_spring_steps(1204.0, 300, &c).snap);
    assert!(std::panic::catch_unwind(|| count_spring_steps(1.0 / 128.0, 300, &config())).is_err());
}

#[test]
fn matrix_powers_equal_repeated_steps_and_the_bound_covers_every_entry() {
    let c = config();
    for constants in CONSTANTS {
        let step = spring_step_matrix(constants, 4.0);
        let (e, v) = euler_steps(constants, 3.0, -7.0, 1);
        assert!((step[0] * 3.0 + step[1] * -7.0 - e).abs() < 1e-12);
        assert!((step[2] * 3.0 + step[3] * -7.0 - v).abs() < 1e-12);
        let bound = spring_matrix_bound(constants, &c);
        let (mut repeated, mut largest) = (IDENTITY_SPRING_MATRIX, 1f64);
        for steps in 0..=300 {
            if steps > 0 {
                repeated = multiply_spring_matrices(repeated, step);
            }
            for (entry, &expected) in spring_matrix_power(step, steps).iter().zip(&repeated) {
                assert!((entry - expected).abs() < 1e-12);
                assert!((*entry as f32 as f64).abs() <= bound);
                largest = largest.max(entry.abs());
            }
        }
        assert!(bound < largest * (1.0 + 1e-5));
    }
    assert!((spring_matrix_bound(CONSTANTS[1], &c) - 10.36).abs() < 0.005);
    assert!(
        std::panic::catch_unwind(|| spring_matrix_bound(SpringConstants { stiffness: 1e7, damping: 0.0 }, &config()))
            .is_err()
    );
    let bad_grid = SpringClockConfig { time_grid: 60.0, ..config() };
    assert!(std::panic::catch_unwind(|| handwired_rs::timing::validate_spring_clock_config(&bad_grid)).is_err());
}

// ---- advance_frame -----------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
struct TestEvent {
    time_stamp: f64,
    id: u32,
}

type Logged = (EvaluationKind, f64, f64, f64, u32, bool, bool, bool, Vec<u32>);

fn log_entry(evaluation: &PlannedEvaluation<TestEvent>) -> Logged {
    (
        evaluation.kind,
        evaluation.time_ms,
        evaluation.simulated_time_ms,
        evaluation.duration_ms,
        evaluation.steps,
        evaluation.settle_check,
        evaluation.snap,
        evaluation.writes_back,
        evaluation.events.iter().map(|event| event.id).collect(),
    )
}

struct Run {
    log: Vec<Logged>,
    clocks: Vec<SpringClock>,
    clock: SpringClock,
    steps_per_frame: Vec<u32>,
}

fn run_planner(
    frame_times: &[f64],
    events: &[TestEvent],
    write_back: WriteBack,
    mut animating: impl FnMut(&PlannedEvaluation<TestEvent>) -> bool,
) -> Run {
    let mut clock = STOPPED_SPRING_CLOCK;
    let (mut log, mut clocks, mut steps_per_frame) = (Vec::new(), Vec::new(), Vec::new());
    let mut next_event = 0;
    for &frame_time in frame_times {
        let mut arrived = Vec::new();
        while next_event < events.len() && events[next_event].time_stamp <= frame_time {
            arrived.push(events[next_event].clone());
            next_event += 1;
        }
        let mut steps = 0;
        clock = advance_frame(
            clock,
            frame_time,
            &arrived,
            |event| event.time_stamp,
            |evaluation| {
                log.push(log_entry(evaluation));
                steps += evaluation.steps;
                animating(evaluation)
            },
            write_back,
            &config(),
        );
        clocks.push(clock);
        steps_per_frame.push(steps);
    }
    Run { log, clocks, clock, steps_per_frame }
}

/// Each event starts an animation that keeps moving until a snap.
fn moves_until_snap() -> impl FnMut(&PlannedEvaluation<TestEvent>) -> bool {
    let mut moving = false;
    move |evaluation| {
        if evaluation.snap {
            moving = false;
        }
        if !evaluation.events.is_empty() {
            moving = true;
        }
        moving
    }
}

fn frame_times_at(hz: f64, duration_ms: f64, stall: Option<(f64, f64)>) -> Vec<f64> {
    let mut times = Vec::new();
    for frame in 0.. {
        let time = frame as f64 * 1000.0 / hz;
        if time > duration_ms {
            break;
        }
        if let Some((start, length)) = stall
            && time > start
            && time < start + length
        {
            continue;
        }
        times.push(time);
    }
    times
}

use EvaluationKind::{Events, Frame, SettleCheck, Snap};

#[test]
fn a_stopped_clock_starts_at_an_event_and_checks_settling_at_fixed_times() {
    let run = run_planner(
        &[100.0, 180.0, 250.0],
        &[TestEvent { time_stamp: 95.3, id: 1 }],
        WriteBack::EveryEvaluation,
        |evaluation| evaluation.time_ms < 159.0,
    );
    assert_eq!(
        run.log,
        vec![
            (Events, 95.296875, 95.296875, 0.0, 0, true, false, true, vec![1]),
            (Frame, 100.0, 99.296875, 4.703125, 1, false, false, true, vec![]),
            (SettleCheck, 159.296875, 159.296875, 59.296875, 15, true, false, true, vec![]),
            (Frame, 180.0, 180.0, 0.0, 0, true, false, true, vec![]),
            (Frame, 250.0, 250.0, 0.0, 0, true, false, true, vec![]),
        ]
    );
    assert_eq!(
        run.clocks[0],
        SpringClock { animated_until_ms: Some(99.296875), check_origin_ms: 95.296875, last_evaluation_ms: Some(100.0) }
    );
    assert_eq!(
        run.clocks[1],
        SpringClock { animated_until_ms: None, check_origin_ms: 95.296875, last_evaluation_ms: None }
    );
}

#[test]
fn evaluations_that_write_back_are_the_same_at_every_frame_rate() {
    let mut rng = Rng::new(53);
    let mut events = Vec::new();
    let mut time = 100.0;
    for id in 0..20 {
        time += 20.0 + rng.uniform(0.0, 400.0);
        events.push(TestEvent { time_stamp: time, id });
    }
    let duration = time + 1000.0;
    let quantized: Vec<f64> = events.iter().map(|event| quantize_time_ms(event.time_stamp, &config())).collect();
    // An event starts an animation that settles at the first settle check 300 ms or more after the latest event.
    let stops_after_events = |evaluation: &PlannedEvaluation<TestEvent>| -> bool {
        if evaluation.kind == Events || !evaluation.settle_check {
            return true;
        }
        if evaluation.kind == Frame {
            return false;
        }
        let latest =
            quantized.iter().copied().filter(|&event| event <= evaluation.time_ms).fold(f64::NEG_INFINITY, f64::max);
        evaluation.time_ms - latest < 300.0
    };
    let writing = |run: Run| -> Vec<Logged> { run.log.into_iter().filter(|entry| entry.7).collect() };
    let reference = writing(run_planner(
        &frame_times_at(120.0, duration, None),
        &events,
        WriteBack::EventsAndSettleChecks,
        stops_after_events,
    ));
    assert!(reference.iter().filter(|entry| entry.0 == SettleCheck).count() > 20);
    for frame_times in [
        frame_times_at(10.0, duration, None),
        frame_times_at(30.0, duration, None),
        frame_times_at(60.0, duration, None),
        frame_times_at(60.0, duration, Some((1000.0, 500.0))),
    ] {
        let run = run_planner(&frame_times, &events, WriteBack::EventsAndSettleChecks, stops_after_events);
        assert!(run.log.iter().filter(|entry| entry.0 == Frame).all(|entry| !entry.7));
        assert_eq!(writing(run), reference);
        let checks = |log: Vec<Logged>| -> Vec<(EvaluationKind, f64, bool, Vec<u32>)> {
            log.into_iter().filter(|entry| entry.0 != Frame).map(|entry| (entry.0, entry.1, entry.5, entry.8)).collect()
        };
        let every = run_planner(&frame_times, &events, WriteBack::EveryEvaluation, stops_after_events);
        assert_eq!(checks(every.log), checks(reference.clone()));
    }
    for entry in &reference {
        assert!(entry.4 <= if entry.0 == SettleCheck { 16 } else { 300 });
        assert!(!entry.6);
    }
}

#[test]
fn past_the_cap_a_snap_stops_the_clock_and_drops_the_backlog() {
    let frames = [0.0, 1000.0 / 60.0, 2000.0 + 1000.0 / 60.0, 2000.0 + 2000.0 / 60.0];
    let stall =
        run_planner(&frames, &[TestEvent { time_stamp: 0.0, id: 0 }], WriteBack::EveryEvaluation, moves_until_snap());
    let short: Vec<_> = stall.log.iter().map(|e| (e.0, e.1, e.2, e.3, e.4, e.5, e.6)).collect();
    assert_eq!(
        short,
        vec![
            (Events, 0.0, 0.0, 0.0, 0, true, false),
            (Frame, 0.0, 0.0, 0.0, 0, false, false),
            (Frame, 16.671875, 16.0, 16.671875, 4, false, false),
            (Snap, 2016.671875, 2016.671875, 2000.0, 0, true, true),
            (Frame, 2016.671875, 2016.671875, 0.0, 0, true, false),
            (Frame, 2033.328125, 2033.328125, 0.0, 0, true, false),
        ]
    );
    assert_eq!(stall.clocks[2].animated_until_ms, None);
    assert_eq!(stall.steps_per_frame, [0, 4, 0, 0]);
    let still_moving =
        run_planner(&frames[..3], &[TestEvent { time_stamp: 0.0, id: 0 }], WriteBack::EveryEvaluation, |_| true);
    assert_eq!(
        still_moving.clock,
        SpringClock {
            animated_until_ms: Some(2016.671875),
            check_origin_ms: 2016.671875,
            last_evaluation_ms: Some(2016.671875)
        }
    );
}

#[test]
fn an_event_during_a_long_stall_applies_at_its_own_time_after_a_snap() {
    let events = [
        TestEvent { time_stamp: 0.0, id: 0 },
        TestEvent { time_stamp: 1500.0, id: 1 },
        TestEvent { time_stamp: 2950.0, id: 2 },
    ];
    let run = run_planner(
        &[0.0, 1000.0 / 60.0, 3000.0 + 1000.0 / 60.0],
        &events,
        WriteBack::EveryEvaluation,
        moves_until_snap(),
    );
    let short: Vec<_> = run.log[3..].iter().map(|e| (e.0, e.1, e.2, e.4, e.5, e.6, e.8.clone())).collect();
    assert_eq!(
        short,
        vec![
            (Snap, 1500.0, 1500.0, 0, true, true, vec![]),
            (Events, 1500.0, 1500.0, 0, true, false, vec![1]),
            (Snap, 2950.0, 2950.0, 0, true, true, vec![]),
            (Events, 2950.0, 2950.0, 0, true, false, vec![2]),
            (SettleCheck, 3014.0, 3014.0, 16, true, false, vec![]),
            (Frame, 3016.671875, 3014.0, 0, false, false, vec![]),
        ]
    );
    let split = run_planner(
        &[0.0, 1000.0 / 60.0, 1250.0],
        &events[..2]
            .iter()
            .map(|e| if e.id == 1 { TestEvent { time_stamp: 1150.0, id: 1 } } else { e.clone() })
            .collect::<Vec<_>>(),
        WriteBack::EveryEvaluation,
        moves_until_snap(),
    );
    assert_eq!(split.steps_per_frame, [0, 4, 283]);
    let tail: Vec<_> = split.log[split.log.len() - 3..].iter().map(|e| (e.0, e.1, e.4)).collect();
    assert_eq!(tail, vec![(Events, 1150.0, 15), (Snap, 1250.0, 0), (Frame, 1250.0, 0)]);
    let checks: Vec<f64> = split.log.iter().filter(|e| e.0 == SettleCheck).map(|e| e.1).collect();
    assert_eq!(checks, (1..=17).map(|k| 64.0 * k as f64).collect::<Vec<_>>());
}

#[test]
fn late_events_apply_at_the_last_evaluation_and_shared_timestamps_share_one() {
    let c = config();
    let clock = advance_frame(
        STOPPED_SPRING_CLOCK,
        100.0,
        &[TestEvent { time_stamp: 100.0, id: 0 }],
        |e| e.time_stamp,
        |_| true,
        WriteBack::EveryEvaluation,
        &c,
    );
    let mut log = Vec::new();
    let events = [
        TestEvent { time_stamp: 90.0, id: 1 },
        TestEvent { time_stamp: 105.0, id: 2 },
        TestEvent { time_stamp: 130.0, id: 3 },
        TestEvent { time_stamp: 105.001, id: 4 },
    ];
    let clock = advance_frame(
        clock,
        116.6667,
        &events,
        |e| e.time_stamp,
        |evaluation| {
            log.push((
                evaluation.kind,
                evaluation.time_ms,
                evaluation.steps,
                evaluation.events.iter().map(|e| e.id).collect::<Vec<_>>(),
            ));
            true
        },
        WriteBack::EveryEvaluation,
        &c,
    );
    assert_eq!(
        log,
        vec![
            (Events, 100.0, 0, vec![1]),
            (Events, 105.0, 1, vec![2, 4]),
            (Events, 116.671875, 3, vec![3]),
            (Frame, 116.671875, 0, vec![]),
        ]
    );
    assert_eq!(
        clock,
        SpringClock { animated_until_ms: Some(116.0), check_origin_ms: 100.0, last_evaluation_ms: Some(116.671875) }
    );
}

#[test]
fn with_events_and_settle_checks_a_frame_evaluates_a_copy() {
    let c = config();
    let committed = advance_frame(
        STOPPED_SPRING_CLOCK,
        10.0,
        &[TestEvent { time_stamp: 10.0, id: 0 }],
        |e| e.time_stamp,
        |_| true,
        WriteBack::EventsAndSettleChecks,
        &c,
    );
    let mut log = Vec::new();
    let after = advance_frame(
        committed,
        40.0,
        &[],
        |e: &TestEvent| e.time_stamp,
        |evaluation| {
            log.push((evaluation.kind, evaluation.steps, evaluation.writes_back, evaluation.duration_ms));
            true
        },
        WriteBack::EventsAndSettleChecks,
        &c,
    );
    assert_eq!(log, vec![(Frame, 7, false, 30.0)]);
    assert_eq!(after, committed);
}

#[test]
fn a_snap_counts_steps_from_the_last_evaluation_that_wrote_back() {
    let snaps_after_gap = |previous_frame: f64, gap: f64, write_back: WriteBack| -> bool {
        let c = config();
        let mut moving = moves_until_snap();
        let mut clock = advance_frame(
            STOPPED_SPRING_CLOCK,
            0.0,
            &[TestEvent { time_stamp: 0.0, id: 0 }],
            |e| e.time_stamp,
            &mut moving,
            write_back,
            &c,
        );
        clock = advance_frame(clock, previous_frame, &[], |e: &TestEvent| e.time_stamp, &mut moving, write_back, &c);
        let mut snapped = false;
        advance_frame(
            clock,
            previous_frame + gap,
            &[],
            |e: &TestEvent| e.time_stamp,
            |evaluation| {
                snapped |= evaluation.snap;
                moving(evaluation)
            },
            write_back,
            &c,
        );
        snapped
    };
    assert!(!snaps_after_gap(60.0, 1203.0, WriteBack::EveryEvaluation));
    assert!(snaps_after_gap(60.0, 1204.0, WriteBack::EveryEvaluation));
    assert!(!snaps_after_gap(60.0, 1143.0, WriteBack::EventsAndSettleChecks));
    assert!(snaps_after_gap(60.0, 1144.0, WriteBack::EventsAndSettleChecks));
    assert!(!snaps_after_gap(64.0, 1203.0, WriteBack::EventsAndSettleChecks));
    assert!(snaps_after_gap(64.0, 1204.0, WriteBack::EventsAndSettleChecks));
}

// ---- emit_spring --------------------------------------------------------------------------------------------------

#[test]
fn a_settled_explicit_spring_lands_exactly_on_its_destination() {
    // Cells: position, destination, velocity, stiffness, damping, timestep, settle distance, settle speed, their
    // scales, and the unit.
    let mut circuit = Circuit::new(11);
    let ranges = [
        (-1e6, 1e6),
        (-1e6, 1e6),
        (-1e6, 1e6),
        (0.0, 1000.0),
        (0.0, 1000.0),
        (0.0, 1.0),
        (0.0, 1e6),
        (0.0, 1e6),
        (0.0, 1e6),
        (0.0, 1e6),
        (1.0, 1.0),
    ];
    for (cell, &(low, high)) in ranges.iter().enumerate() {
        circuit.assume_range(circuit.source(cell), low, high);
    }
    let cell = |index| Wire::from_index(index);
    let spring = handwired_rs::emit_spring(
        &mut circuit,
        cell(0),
        cell(1),
        cell(2),
        cell(3),
        cell(4),
        cell(5),
        cell(6),
        cell(7),
        cell(8),
        cell(9),
        cell(10),
    );
    let program = circuit.compile(Some(&[spring.position, spring.velocity, spring.active]));
    for (start, destination) in [(100.0f32, 0.0f32), (5.0, 42.0), (-30.0, -7.0), (0.0, 250.0)] {
        let (mut position, mut velocity, mut active) = (start, 0.0f32, 1.0f32);
        for _ in 0..240 {
            let source = [position, destination, velocity, 200.0, 20.0, 0.05, 0.5, 0.5, 1000.0, 1000.0, 1.0];
            let reference = circuit.evaluate_reference(&source).unwrap();
            let tensor = circuit.evaluate_tensor(&source, &program).unwrap();
            for wire in [spring.position, spring.velocity, spring.active] {
                assert_eq!(tensor[wire.index()].to_bits(), reference[wire.index()].to_bits());
            }
            (position, velocity, active) = (
                reference[spring.position.index()],
                reference[spring.velocity.index()],
                reference[spring.active.index()],
            );
        }
        assert_eq!((active, position, velocity), (0.0, destination, 0.0));
    }
}

// ---- Springs along frame-time traces -------------------------------------------------------------------------------

// A small app along frame-time traces. Object 0 has x and y springs (stiffness 290, damping 30) and a scale spring
// (stiffness 333, damping 20); object 1 has one spring (stiffness 290, damping 30). A press replaces one object's
// destinations, and a jump press also snaps that object onto them. The host writes one matrix per stiffness and
// damping pair before each evaluation, and 1 into every object's snap cell for a planned snap.

#[derive(Clone, Debug)]
struct Press {
    time_stamp: f64,
    object: usize,
    destinations: Vec<f64>,
    jump: bool,
}

const APP_PAIRS: [SpringConstants; 2] =
    [SpringConstants { stiffness: 290.0, damping: 30.0 }, SpringConstants { stiffness: 333.0, damping: 20.0 }];
const APP_SPRING_PAIR: [usize; 4] = [0, 0, 1, 0];
const APP_OBJECT_SPRINGS: [&[usize]; 2] = [&[0, 1, 2], &[3]];
const APP_INITIAL_POSITIONS: [f64; 4] = [400.0, 300.0, 1.0, 200.0];
const APP_SETTLE_CHECK: usize = 8;
const APP_EVENT_ACTIVE: usize = 9;
const APP_SNAP: usize = 11;
const APP_EVENT_DESTINATION: usize = 13;
const APP_SIZE: usize = 29;
const APP_POSITIONS: usize = APP_SIZE;
const APP_ACTIVE: usize = APP_SIZE + 4;
const APP_SETTLED: usize = APP_SIZE + 6;

/// Per spring: its displacement from the destination in effect (field 0), its velocity (1), and that destination (2).
fn app_spring_cell(spring: usize, field: usize) -> usize {
    17 + 3 * spring + field
}

fn app_object_of(spring: usize) -> usize {
    APP_OBJECT_SPRINGS.iter().position(|members| members.contains(&spring)).unwrap()
}

fn app_program() -> TensorProgram {
    let mut circuit = Circuit::new(APP_SIZE);
    for (pair, &constants) in APP_PAIRS.iter().enumerate() {
        let bound = spring_matrix_bound(constants, &config());
        for entry in 0..4 {
            circuit.assume_range(circuit.source(4 * pair + entry), -bound, bound);
        }
    }
    circuit.assume_range(circuit.source(APP_SETTLE_CHECK), 0.0, 1.0);
    for object in 0..2 {
        circuit.assume_range(circuit.source(APP_EVENT_ACTIVE + object), 0.0, 1.0);
        circuit.assume_range(circuit.source(APP_SNAP + object), 0.0, 1.0);
    }
    let mut destinations = Vec::new();
    for spring in 0..4 {
        let active = circuit.source(APP_EVENT_ACTIVE + app_object_of(spring));
        let (current, next) =
            (circuit.source(app_spring_cell(spring, 2)), circuit.source(APP_EVENT_DESTINATION + spring));
        destinations.push(circuit.select(active, current, next));
    }
    let (mut positions, mut displacements, mut velocities) =
        ([Wire::from_index(0); 4], [Wire::from_index(0); 4], [Wire::from_index(0); 4]);
    let mut objects = Vec::new();
    for (object, members) in APP_OBJECT_SPRINGS.iter().enumerate() {
        let mut springs = Vec::new();
        for &spring in members.iter() {
            let start = 4 * APP_SPRING_PAIR[spring];
            springs.push(AnimatedSpring {
                matrix: [
                    circuit.source(start),
                    circuit.source(start + 1),
                    circuit.source(start + 2),
                    circuit.source(start + 3),
                ],
                displacement: circuit.source(app_spring_cell(spring, 0)),
                velocity: circuit.source(app_spring_cell(spring, 1)),
                destination: destinations[spring],
                destination_during_steps: Some(circuit.source(app_spring_cell(spring, 2))),
            });
        }
        let mut options = AnimatedObjectOptions::new(circuit.source(APP_SETTLE_CHECK));
        options.snap = Some(circuit.source(APP_SNAP + object));
        let emitted = emit_animated_object(&mut circuit, &springs, options);
        for (index, &spring) in members.iter().enumerate() {
            positions[spring] = emitted.positions[index];
            displacements[spring] = emitted.displacements[index];
            velocities[spring] = emitted.velocities[index];
        }
        objects.push(emitted);
    }
    let zero = circuit.constant(0.0);
    let mut roots: Vec<Wire> = (0..APP_SIZE).map(|cell| circuit.source(cell)).collect();
    for object in 0..2 {
        roots[APP_EVENT_ACTIVE + object] = zero;
        roots[APP_SNAP + object] = zero;
    }
    for spring in 0..4 {
        roots[APP_EVENT_DESTINATION + spring] = zero;
        roots[app_spring_cell(spring, 0)] = displacements[spring];
        roots[app_spring_cell(spring, 1)] = velocities[spring];
        roots[app_spring_cell(spring, 2)] = destinations[spring];
    }
    roots.extend(positions);
    roots.extend(objects.iter().map(|object| object.active));
    roots.extend(objects.iter().map(|object| object.settled));
    circuit.compile(Some(&roots))
}

struct Drawn {
    positions: [f32; 4],
    outputs: Vec<f32>,
}

type Settle = (f64, [f32; 2]);

struct AppRun {
    frames: HashMap<u64, Drawn>,
    settles: Vec<Settle>,
    commits: Vec<(f64, EvaluationKind, Vec<f32>)>,
    steps_per_frame: Vec<u32>,
    clock: SpringClock,
    /// Evaluations that wrote state back without checking settling, and found nothing animating.
    stops_without_settle_check: usize,
}

/// `offset` is added to every destination, e.g. 65,536 runs the app near 65,536 px.
fn run_app(
    program: &TensorProgram,
    presses: &[Press],
    frame_times: &[f64],
    write_back: WriteBack,
    offset: f64,
) -> AppRun {
    let mut executor = TensorExecutor::new(program);
    let config = config();
    let powers: Vec<Vec<[f32; 4]>> = APP_PAIRS
        .iter()
        .map(|&constants| {
            let step = spring_step_matrix(constants, config.step_ms);
            (0..=config.max_steps_per_frame)
                .map(|steps| spring_matrix_power(step, steps).map(|entry| entry as f32))
                .collect()
        })
        .collect();
    let mut state = vec![0f32; APP_SIZE];
    for (spring, position) in APP_INITIAL_POSITIONS.iter().enumerate() {
        state[app_spring_cell(spring, 2)] = (position + offset) as f32;
    }
    let mut scratch = vec![0f32; APP_SIZE];
    let mut run = AppRun {
        frames: HashMap::new(),
        settles: Vec::new(),
        commits: Vec::new(),
        steps_per_frame: Vec::new(),
        clock: STOPPED_SPRING_CLOCK,
        stops_without_settle_check: 0,
    };
    let mut next_press = 0;
    for &frame_time in frame_times {
        let mut arrived = Vec::new();
        while next_press < presses.len() && presses[next_press].time_stamp <= frame_time {
            arrived.push(presses[next_press].clone());
            next_press += 1;
        }
        let mut steps = 0;
        run.clock = advance_frame(
            run.clock,
            frame_time,
            &arrived,
            |press| press.time_stamp,
            |evaluation| {
                steps += evaluation.steps;
                let table = if evaluation.writes_back {
                    &mut state
                } else {
                    scratch.copy_from_slice(&state);
                    &mut scratch
                };
                for (pair, pair_powers) in powers.iter().enumerate() {
                    table[4 * pair..4 * pair + 4].copy_from_slice(&pair_powers[evaluation.steps as usize]);
                }
                table[APP_SETTLE_CHECK] = if evaluation.settle_check { 1.0 } else { 0.0 };
                for object in 0..2 {
                    table[APP_SNAP + object] = if evaluation.snap { 1.0 } else { 0.0 };
                }
                for press in &evaluation.events {
                    table[APP_EVENT_ACTIVE + press.object] = 1.0;
                    if press.jump {
                        table[APP_SNAP + press.object] = 1.0;
                    }
                    for (index, &spring) in APP_OBJECT_SPRINGS[press.object].iter().enumerate() {
                        table[APP_EVENT_DESTINATION + spring] = (press.destinations[index] + offset) as f32;
                    }
                }
                let outputs = executor.run(table).to_vec();
                if evaluation.writes_back {
                    state.copy_from_slice(&outputs[..APP_SIZE]);
                    run.commits.push((evaluation.time_ms, evaluation.kind, state.clone()));
                    if evaluation.settle_check && evaluation.kind != Frame {
                        run.settles.push((evaluation.time_ms, [outputs[APP_SETTLED], outputs[APP_SETTLED + 1]]));
                    }
                }
                if evaluation.kind == Frame {
                    let positions = [0, 1, 2, 3].map(|spring| outputs[APP_POSITIONS + spring]);
                    run.frames.insert(frame_time.to_bits(), Drawn { positions, outputs: outputs.clone() });
                }
                let animating = outputs[APP_ACTIVE] > 0.0 || outputs[APP_ACTIVE + 1] > 0.0;
                if evaluation.writes_back && !evaluation.settle_check && !animating {
                    run.stops_without_settle_check += 1;
                }
                animating
            },
            write_back,
            &config,
        );
        run.steps_per_frame.push(steps);
    }
    run
}

#[derive(Clone, Copy)]
struct Float64Spring {
    position: f64,
    velocity: f64,
    destination: f64,
}

fn step_float64_springs(springs: &mut [Float64Spring]) {
    for (index, spring) in springs.iter_mut().enumerate() {
        let SpringConstants { stiffness, damping } = APP_PAIRS[APP_SPRING_PAIR[index]];
        let displacement = spring.position - spring.destination;
        spring.velocity += (-stiffness * displacement - damping * spring.velocity) * 0.004;
        spring.position = spring.destination + displacement + spring.velocity * 0.004;
    }
}

/// Snaps every object whose springs are all within both thresholds, and returns the bits.
fn settle_float64_objects(springs: &mut [Float64Spring]) -> [f32; 2] {
    let threshold = 0.01f32 as f64;
    let mut bits = [0f32; 2];
    for (object, members) in APP_OBJECT_SPRINGS.iter().enumerate() {
        let settled = members.iter().all(|&spring| {
            (springs[spring].position - springs[spring].destination).abs() <= threshold
                && springs[spring].velocity.abs() <= threshold
        });
        if settled {
            for &spring in members.iter() {
                springs[spring].position = springs[spring].destination;
                springs[spring].velocity = 0.0;
            }
        }
        bits[object] = if settled { 1.0 } else { 0.0 };
    }
    bits
}

type Traces = HashMap<u64, [f64; 4]>;

struct Float64Run {
    positions: Traces,
    displacements: Traces,
    settles: Vec<Settle>,
}

/// Frame-free float64 springs: 4 ms Euler steps from the moment the clock started, settle checks every 16 steps,
/// presses applied at their timestamps after stepping to them, and a stopped clock that starts at a press. A jump
/// snaps its object after the press's settle check, and the clock stops once every spring is exactly at rest.
fn run_float64_reference(presses: &[Press], frame_times: &[f64], offset: f64) -> Float64Run {
    let mut springs = APP_INITIAL_POSITIONS.map(|position| Float64Spring {
        position: position + offset,
        velocity: 0.0,
        destination: position + offset,
    });
    let (mut animated_until_ms, mut origin_ms): (Option<f64>, f64) = (None, 0.0);
    let mut run = Float64Run { positions: HashMap::new(), displacements: HashMap::new(), settles: Vec::new() };
    let advance_to = |springs: &mut [Float64Spring; 4],
                      animated_until_ms: &mut Option<f64>,
                      origin_ms: f64,
                      settles: &mut Vec<Settle>,
                      time_ms: f64| {
        while let Some(until) = *animated_until_ms
            && until + 4.0 <= time_ms
        {
            step_float64_springs(springs);
            let stepped = until + 4.0;
            let stops = (stepped - origin_ms) % 64.0 == 0.0 && {
                let settled = settle_float64_objects(springs);
                settles.push((stepped, settled));
                settled.iter().all(|&bit| bit == 1.0)
            };
            *animated_until_ms = if stops { None } else { Some(stepped) };
        }
    };
    let mut next_press = 0;
    for &frame_time in frame_times {
        while next_press < presses.len() && presses[next_press].time_stamp <= frame_time {
            let time_ms = presses[next_press].time_stamp;
            advance_to(&mut springs, &mut animated_until_ms, origin_ms, &mut run.settles, time_ms);
            let mut jumped = Vec::new();
            while next_press < presses.len() && presses[next_press].time_stamp == time_ms {
                let press = &presses[next_press];
                next_press += 1;
                for (index, &spring) in APP_OBJECT_SPRINGS[press.object].iter().enumerate() {
                    springs[spring].destination = press.destinations[index] + offset;
                }
                if press.jump {
                    jumped.push(press.object);
                }
            }
            let snap_jumped = |springs: &mut [Float64Spring; 4]| {
                for &object in &jumped {
                    for &spring in APP_OBJECT_SPRINGS[object] {
                        springs[spring].position = springs[spring].destination;
                        springs[spring].velocity = 0.0;
                    }
                }
            };
            if animated_until_ms.is_none() {
                let settled = settle_float64_objects(&mut springs);
                run.settles.push((time_ms, settled));
                snap_jumped(&mut springs);
                if settled.iter().enumerate().any(|(object, &bit)| bit == 0.0 && !jumped.contains(&object)) {
                    animated_until_ms = Some(time_ms);
                    origin_ms = time_ms;
                }
            } else {
                snap_jumped(&mut springs);
                if springs.iter().all(|spring| spring.position == spring.destination && spring.velocity == 0.0) {
                    animated_until_ms = None;
                }
            }
        }
        advance_to(&mut springs, &mut animated_until_ms, origin_ms, &mut run.settles, frame_time);
        run.positions.insert(frame_time.to_bits(), springs.map(|spring| spring.position));
        run.displacements.insert(frame_time.to_bits(), springs.map(|spring| spring.position - spring.destination));
    }
    run
}

/// chenguini ui.md's order in float64: handle inputs, new layout, then an animation tick that catches up toward the
/// new destinations, with one settle check per frame.
fn run_ui_md_order(presses: &[Press], frame_times: &[f64]) -> Traces {
    let mut springs =
        APP_INITIAL_POSITIONS.map(|position| Float64Spring { position, velocity: 0.0, destination: position });
    let mut animated_until_ms: Option<f64> = None;
    let mut positions = HashMap::new();
    let mut next_press = 0;
    for &now in frame_times {
        while next_press < presses.len() && presses[next_press].time_stamp <= now {
            let press = &presses[next_press];
            next_press += 1;
            for (index, &spring) in APP_OBJECT_SPRINGS[press.object].iter().enumerate() {
                springs[spring].destination = press.destinations[index];
            }
        }
        let mut until = animated_until_ms.unwrap_or(now);
        let steps = 300.0f64.min(((now - until) / 4.0).floor());
        until += steps * 4.0;
        for _ in 0..steps as u32 {
            step_float64_springs(&mut springs);
        }
        animated_until_ms =
            if settle_float64_objects(&mut springs).iter().all(|&bit| bit == 1.0) { None } else { Some(until) };
        positions.insert(now.to_bits(), springs.map(|spring| spring.position));
    }
    positions
}

/// handwired's test generator (mulberry32), so the traces are the same presses.
fn mulberry32(mut seed: u32) -> impl FnMut() -> f64 {
    move || {
        seed = seed.wrapping_add(0x6D2B_79F5);
        let mut value = (seed ^ (seed >> 15)).wrapping_mul(1 | seed);
        value = value.wrapping_add((value ^ (value >> 7)).wrapping_mul(61 | value)) ^ value;
        (value ^ (value >> 14)) as f64 / 4_294_967_296.0
    }
}

/// 12 presses between frames on the 1/64 ms grid, 90 to 700 ms apart, so most retarget moving springs. With
/// `jump_chance` above 0, some presses are jumps.
fn app_presses(seed: u32, jump_chance: f64) -> Vec<Press> {
    let mut random = mulberry32(seed);
    let mut presses = Vec::new();
    let mut time = 100.0;
    for _ in 0..12 {
        time += 90.0 + random() * 610.0;
        let object = if random() < 0.6 { 0 } else { 1 };
        let destinations = if object == 0 {
            let x = (random() * 800.0).round();
            let y = (random() * 600.0).round();
            vec![x, y, ((0.5 + random()) * 64.0).round() / 64.0]
        } else {
            vec![(random() * 800.0).round()]
        };
        // Drawn only when asked, so the traces without jumps stay the same.
        let jump = jump_chance > 0.0 && random() < jump_chance;
        presses.push(Press { time_stamp: (time * 64.0).round() / 64.0, object, destinations, jump });
    }
    presses
}

fn frame_positions(run: &AppRun) -> Traces {
    run.frames.iter().map(|(&time, frame)| (time, frame.positions.map(f64::from))).collect()
}

/// Each spring's stored displacement after each frame's evaluation.
fn frame_displacements(run: &AppRun) -> Traces {
    run.frames
        .iter()
        .map(|(&time, frame)| (time, [0, 1, 2, 3].map(|spring| frame.outputs[app_spring_cell(spring, 0)] as f64)))
        .collect()
}

fn max_frame_difference(left: &Traces, right: &Traces) -> f64 {
    let (mut worst, mut shared) = (0f64, 0);
    for (time, positions) in left {
        let Some(other) = right.get(time) else { continue };
        shared += 1;
        for (position, other) in positions.iter().zip(other) {
            worst = worst.max((position - other).abs());
        }
    }
    assert!(shared > 20, "only {shared} shared frames");
    worst
}

fn differing_bits(left: &[f32], right: &[f32]) -> usize {
    left.iter().zip(right).filter(|(left, right)| left.to_bits() != right.to_bits()).count()
}

/// Seeds 1 to 8 without jumps, then 4 seeds where about a third of the presses jump.
fn trace_cases() -> Vec<(u32, f64)> {
    (1..=8).map(|seed| (seed, 0.0)).chain((21..=24).map(|seed| (seed, 0.35))).collect()
}

/// The 120 fps baseline, then 10, 30 and 60 fps and 500 ms stalls at 60 and 120 fps.
fn trace_schedules(presses: &[Press]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let duration_ms = presses.last().unwrap().time_stamp + 2000.0;
    let stall = Some((presses[2].time_stamp - 100.0, 500.0));
    let others = vec![
        frame_times_at(10.0, duration_ms, None),
        frame_times_at(30.0, duration_ms, None),
        frame_times_at(60.0, duration_ms, None),
        frame_times_at(60.0, duration_ms, stall),
        frame_times_at(120.0, duration_ms, stall),
    ];
    (frame_times_at(120.0, duration_ms, None), others)
}

fn commit_log(run: &AppRun) -> Vec<(EvaluationKind, f64)> {
    run.commits.iter().map(|&(time, kind, _)| (kind, time)).collect()
}

#[test]
fn along_frame_traces_committed_tables_and_displays_are_bit_identical_at_every_frame_rate() {
    let program = app_program();
    let mut stops_at_jumps = 0;
    for (seed, jump_chance) in trace_cases() {
        let presses = app_presses(seed, jump_chance);
        let (baseline, others) = trace_schedules(&presses);
        let baseline_run = run_app(&program, &presses, &baseline, WriteBack::EventsAndSettleChecks, 0.0);
        for frame_times in std::iter::once(&baseline).chain(&others) {
            let run = run_app(&program, &presses, frame_times, WriteBack::EventsAndSettleChecks, 0.0);
            assert_eq!(commit_log(&run), commit_log(&baseline_run), "seed {seed}");
            for (commit, baseline_commit) in run.commits.iter().zip(&baseline_run.commits) {
                assert_eq!(differing_bits(&commit.2, &baseline_commit.2), 0, "seed {seed} at {}", commit.0);
            }
            let mut shared_displays = 0;
            for (time, frame) in &run.frames {
                let Some(other) = baseline_run.frames.get(time) else { continue };
                shared_displays += 1;
                assert_eq!(
                    differing_bits(&frame.outputs, &other.outputs),
                    0,
                    "seed {seed} at {}",
                    f64::from_bits(*time)
                );
            }
            assert!(shared_displays > 20);
            let reference = run_float64_reference(&presses, frame_times, 0.0);
            assert_eq!(run.settles, reference.settles, "seed {seed}");
            assert!(max_frame_difference(&frame_positions(&run), &reference.positions) < 5e-4, "seed {seed}");
            assert!(run.commits.iter().all(|commit| commit.1 != Snap));
        }
        // Without jumps, a running clock only stops at a settle check. A jump snaps its object, and when the other
        // object is exactly at rest, nothing animates and the clock stops at the jump, which needs the rest bit.
        if jump_chance == 0.0 {
            assert_eq!(baseline_run.stops_without_settle_check, 0, "seed {seed}");
        } else {
            stops_at_jumps += baseline_run.stops_without_settle_check;
        }
    }
    // handwired measured 5 stops over the 4 traces with jumps.
    assert!(stops_at_jumps > 0);
}

#[test]
fn writing_back_every_evaluation_stays_within_5e_4_px_of_float64_and_settles_like_it() {
    let program = app_program();
    let mut runs_with_different_settles = 0;
    for (seed, jump_chance) in trace_cases() {
        let presses = app_presses(seed, jump_chance);
        let (baseline, others) = trace_schedules(&presses);
        let baseline_run = run_app(&program, &presses, &baseline, WriteBack::EveryEvaluation, 0.0);
        for frame_times in std::iter::once(&baseline).chain(&others) {
            let run = run_app(&program, &presses, frame_times, WriteBack::EveryEvaluation, 0.0);
            let reference = run_float64_reference(&presses, frame_times, 0.0);
            assert!(max_frame_difference(&frame_positions(&run), &reference.positions) < 5e-4, "seed {seed}");
            assert!(
                max_frame_difference(&frame_positions(&run), &frame_positions(&baseline_run)) < 5e-4,
                "seed {seed}"
            );
            if run.settles != reference.settles {
                runs_with_different_settles += 1;
            }
        }
    }
    // Every evaluation rounds the stored displacement at its own size, so drift shrinks near rest.
    assert_eq!(runs_with_different_settles, 0);
}

#[test]
fn a_3_s_stall_snaps_every_object_once_and_tables_stay_bit_identical_across_frame_rates() {
    let program = app_program();
    let press = |time_stamp: f64, object: usize, destinations: &[f64]| Press {
        time_stamp,
        object,
        destinations: destinations.to_vec(),
        jump: false,
    };
    let presses = [
        press(100.0, 0, &[700.0, 50.0, 1.5]),
        press(250.0, 1, &[20.0]),
        // During the stall, 2.9 s before the next frame.
        press(500.0, 0, &[120.0, 480.0, 0.75]),
        press(4000.0, 1, &[640.0]),
        press(4100.0, 0, &[300.0, 300.0, 1.0]),
    ];
    let stall = Some((400.0, 3000.0));
    let runs: Vec<AppRun> = [120.0, 60.0, 30.0, 10.0]
        .iter()
        .map(|&hz| {
            run_app(&program, &presses, &frame_times_at(hz, 6000.0, stall), WriteBack::EventsAndSettleChecks, 0.0)
        })
        .collect();
    for run in &runs {
        assert!(run.steps_per_frame.iter().all(|&steps| steps <= 300));
        let snaps: Vec<&(f64, EvaluationKind, Vec<f32>)> =
            run.commits.iter().filter(|commit| commit.1 == Snap).collect();
        assert_eq!(snaps.iter().map(|commit| commit.0).collect::<Vec<_>>(), vec![3400.0]);
        // Right after the snap every displacement and velocity is ±0.
        for spring in 0..4 {
            assert_eq!(snaps[0].2[app_spring_cell(spring, 0)], 0.0);
            assert_eq!(snaps[0].2[app_spring_cell(spring, 1)], 0.0);
        }
        assert_eq!(commit_log(run), commit_log(&runs[0]));
        for (commit, first) in run.commits.iter().zip(&runs[0].commits) {
            assert_eq!(differing_bits(&commit.2, &first.2), 0);
        }
        assert_eq!(run.clock.animated_until_ms, None);
    }
    // Every frame rate resumes at 3,400 ms, and shows the snapped objects there.
    assert_eq!(runs[3].frames[&3400f64.to_bits()].positions, [120.0, 480.0, 0.75, 20.0]);
}

#[test]
fn far_from_the_origin_both_policies_settle_like_float64_and_displacements_stay_close() {
    let program = app_program();
    for offset in [65_536.0, 131_072.0, 1_048_576.0f64] {
        // FP32 values near the offset are this far apart, e.g. 1/128 px near 65,536 px.
        let spacing = 2f64.powf(offset.log2() - 23.0);
        for write_back in [WriteBack::EveryEvaluation, WriteBack::EventsAndSettleChecks] {
            let (mut worst_displacement, mut worst_position) = (0f64, 0f64);
            for seed in 1..=8 {
                let presses = app_presses(seed, 0.0);
                let (baseline, others) = trace_schedules(&presses);
                for frame_times in std::iter::once(&baseline).chain(&others) {
                    let run = run_app(&program, &presses, frame_times, write_back, offset);
                    let reference = run_float64_reference(&presses, frame_times, offset);
                    assert_eq!(run.settles, reference.settles, "offset {offset} seed {seed}");
                    // Nothing still animates 2 s after the last press.
                    assert_eq!(run.clock.animated_until_ms, None);
                    worst_displacement = worst_displacement
                        .max(max_frame_difference(&frame_displacements(&run), &reference.displacements));
                    worst_position =
                        worst_position.max(max_frame_difference(&frame_positions(&run), &reference.positions));
                }
            }
            if offset <= 131_072.0 {
                // handwired measured 1.4e-4 px writing back every evaluation and 7.4e-5 px with events and settle
                // checks, as at 0 px.
                assert!(worst_displacement < 5e-4, "{offset} {write_back:?}: {worst_displacement}");
                // The drawn position rounds once at the destination's magnitude.
                assert!(worst_position < spacing / 2.0 + 5e-4, "{offset} {write_back:?}: {worst_position}");
            } else {
                // FP32 values near 1,048,576 px are 1/8 px apart, so the scale spring's destinations, on a 1/64 grid,
                // round before they reach the program.
                assert!(worst_displacement < spacing, "{offset} {write_back:?}: {worst_displacement}");
            }
        }
    }
}

#[test]
fn ui_md_order_applying_presses_then_catching_up_differs_across_frame_rates() {
    let (mut worst_10, mut worst_30, mut worst_60) = (0f64, 0f64, 0f64);
    for seed in 1..=8 {
        let presses = app_presses(seed, 0.0);
        let duration_ms = presses.last().unwrap().time_stamp + 2000.0;
        let baseline = run_ui_md_order(&presses, &frame_times_at(120.0, duration_ms, None));
        let at = |hz: f64| run_ui_md_order(&presses, &frame_times_at(hz, duration_ms, None));
        worst_10 = worst_10.max(max_frame_difference(&at(10.0), &baseline));
        worst_30 = worst_30.max(max_frame_difference(&at(30.0), &baseline));
        worst_60 = worst_60.max(max_frame_difference(&at(60.0), &baseline));
    }
    // handwired measured 359, 124 and 53 px.
    assert!(worst_10 > 100.0 && worst_30 > 50.0 && worst_60 > 10.0, "{worst_10} {worst_30} {worst_60}");
}
