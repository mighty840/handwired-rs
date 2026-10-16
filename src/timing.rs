//! Host timing for springs whose step matrix the host writes into source cells,
//! ported from handwired's `host-timing.ts`.
//!
//! Everything here is plain f64 code that runs between evaluations and never
//! builds operations. It decides what the host writes before each evaluation:
//! times on a 1/64 ms grid, how many fixed 4 ms spring steps an evaluation
//! runs, the spring matrix for that step count, when to evaluate between two
//! frames so springs step to each event's timestamp before the event applies,
//! settle checks at fixed simulated times, and when to snap every animated
//! object onto its destination because a frame would need more steps than its
//! cap. Bad arguments panic, like the builder's contract checks.

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SpringClockConfig {
    /// Times are rounded to multiples of 1 / time_grid ms, e.g. 64 for a 1/64 ms grid.
    pub time_grid: f64,
    /// One spring step simulates this many milliseconds.
    pub step_ms: f64,
    /// The most steps the evaluations of one frame run together; past it, objects snap.
    pub max_steps_per_frame: u32,
    /// Settling is checked every this many steps of simulated time, from when the clock started.
    pub settle_check_steps: u32,
}

impl Default for SpringClockConfig {
    fn default() -> SpringClockConfig {
        SpringClockConfig { time_grid: 64.0, step_ms: 4.0, max_steps_per_frame: 300, settle_check_steps: 16 }
    }
}

pub fn validate_spring_clock_config(config: &SpringClockConfig) {
    let SpringClockConfig { time_grid, step_ms, max_steps_per_frame, settle_check_steps } = *config;
    assert!(
        time_grid >= 1.0 && time_grid.fract() == 0.0 && time_grid.log2().fract() == 0.0,
        "the time grid must be a whole power of two, got {time_grid}"
    );
    assert!(
        step_ms > 0.0 && (step_ms * time_grid).fract() == 0.0,
        "a spring step must be a positive multiple of the time grid"
    );
    assert!(max_steps_per_frame >= 1, "max_steps_per_frame must be a positive integer");
    assert!(settle_check_steps >= 1, "settle_check_steps must be a positive integer");
    // Durations and event times are multiples of the grid within the cap, exact in FP32 within 2^24 grid steps.
    assert!(
        max_steps_per_frame as f64 * step_ms * time_grid <= 16_777_216.0,
        "max_steps_per_frame × step_ms must stay within 2^24 grid steps"
    );
}

/// A timestamp rounded to the time grid, e.g. 16.6667 ms becomes 16.671875 ms on the 1/64 ms grid. Quantizing
/// absolute timestamps before subtracting makes the durations of any split of a span add up. Never -0.
pub fn quantize_time_ms(time_stamp: f64, config: &SpringClockConfig) -> f64 {
    assert!(time_stamp.is_finite(), "a timestamp must be finite, got {time_stamp}");
    js_round(time_stamp * config.time_grid) / config.time_grid + 0.0
}

/// JavaScript's `Math.round`: the nearest integer, halves toward +∞ (Rust's `round` takes halves away from zero).
fn js_round(value: f64) -> f64 {
    let rounded = value.round();
    if rounded - value == -0.5 { rounded + 1.0 } else { rounded }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FrameTiming {
    /// The frame's own timestamp on the grid, never before the previous frame.
    pub frame_time_ms: f64,
    /// Time since the previous frame, at most the cap, or 0 for the first frame after idle.
    pub duration_ms: f64,
    /// Each event's time relative to the frame, e.g. -3.5 for a press 3.5 ms before it, clamped to 0 for an event
    /// after the frame and to the later of the previous frame and -cap for an older one.
    pub event_times_ms: Vec<f64>,
}

/// What the host writes into a program's time cells for one frame: the frame's duration and its events' times
/// relative to the frame, on the time grid. Absolute timestamps stay in f64, since an FP32 cell can't hold them.
pub fn quantize_frame_timing(
    frame_time_stamp: f64,
    previous_frame_time_ms: Option<f64>,
    event_time_stamps: &[f64],
    config: &SpringClockConfig,
) -> FrameTiming {
    validate_spring_clock_config(config);
    let cap_ms = config.max_steps_per_frame as f64 * config.step_ms;
    let mut frame_time_ms = quantize_time_ms(frame_time_stamp, config);
    if let Some(previous) = previous_frame_time_ms {
        frame_time_ms = frame_time_ms.max(previous);
    }
    let duration_ms = previous_frame_time_ms.map_or(0.0, |previous| cap_ms.min(frame_time_ms - previous));
    let earliest_event_ms = previous_frame_time_ms.map_or(-cap_ms, |previous| (-cap_ms).max(previous - frame_time_ms));
    let event_times_ms = event_time_stamps
        .iter()
        .map(|&time_stamp| 0f64.min(earliest_event_ms.max(quantize_time_ms(time_stamp, config) - frame_time_ms)) + 0.0)
        .collect();
    FrameTiming { frame_time_ms, duration_ms, event_times_ms }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct StepCount {
    /// floor(pending / step_ms), or 0 when snapping.
    pub steps: u32,
    /// Pending time these steps don't simulate, less than one step, carried into a later evaluation.
    pub leftover_ms: f64,
    /// More steps are pending than the budget: snap every object onto its destination and drop the pending time.
    pub snap: bool,
}

/// How many steps an evaluation runs: `pending_ms` is the time since the springs last stepped, on the time grid,
/// and `step_budget` what's left of the frame's cap. E.g. 17.5 ms pending runs 4 steps and carries 1.5 ms.
pub fn count_spring_steps(pending_ms: f64, step_budget: u32, config: &SpringClockConfig) -> StepCount {
    assert!(
        (pending_ms * config.time_grid).fract() == 0.0 && pending_ms >= 0.0,
        "pending time must lie on the time grid and be nonnegative, got {pending_ms}"
    );
    let steps = (pending_ms / config.step_ms).floor();
    if steps > step_budget as f64 {
        return StepCount { steps: 0, leftover_ms: 0.0, snap: true };
    }
    StepCount { steps: steps as u32, leftover_ms: pending_ms - steps * config.step_ms, snap: false }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SpringConstants {
    pub stiffness: f64,
    pub damping: f64,
}

/// `(displacement, velocity)` after the steps: `[d←d, d←v, v←d, v←v]`.
pub type SpringMatrix = [f64; 4];

pub const IDENTITY_SPRING_MATRIX: SpringMatrix = [1.0, 0.0, 0.0, 1.0];

/// One semi-implicit Euler step as a matrix: `velocity' = velocity + (-stiffness × displacement - damping ×
/// velocity) × dt`, `displacement' = displacement + velocity' × dt`.
pub fn spring_step_matrix(constants: SpringConstants, step_ms: f64) -> SpringMatrix {
    assert!(
        constants.stiffness.is_finite()
            && constants.stiffness >= 0.0
            && constants.damping.is_finite()
            && constants.damping >= 0.0,
        "stiffness and damping must be finite and nonnegative"
    );
    let dt = step_ms / 1000.0;
    let velocity_keep = 1.0 - constants.damping * dt;
    [1.0 - constants.stiffness * dt * dt, dt * velocity_keep, -constants.stiffness * dt, velocity_keep]
}

pub fn multiply_spring_matrices(left: SpringMatrix, right: SpringMatrix) -> SpringMatrix {
    [
        left[0] * right[0] + left[1] * right[2],
        left[0] * right[1] + left[1] * right[3],
        left[2] * right[0] + left[3] * right[2],
        left[2] * right[1] + left[3] * right[3],
    ]
}

/// `matrix^count` by repeated squaring, e.g. 4 steps' matrix is `spring_matrix_power(step, 4)`.
pub fn spring_matrix_power(matrix: SpringMatrix, count: u32) -> SpringMatrix {
    let (mut result, mut square, mut remaining) = (IDENTITY_SPRING_MATRIX, matrix, count);
    while remaining > 0 {
        if remaining % 2 == 1 {
            result = multiply_spring_matrices(result, square);
        }
        remaining /= 2;
        if remaining > 0 {
            square = multiply_spring_matrices(square, square);
        }
    }
    result
}

/// The magnitude to declare on the 4 matrix cells: the largest |entry| of M^n for n = 0..=max_steps_per_frame,
/// with a little headroom so every FP32 entry the host writes stays within it, e.g. 10.4 for stiffness 650 and
/// damping 45. Panics for a spring whose powers overflow.
pub fn spring_matrix_bound(constants: SpringConstants, config: &SpringClockConfig) -> f64 {
    let step = spring_step_matrix(constants, config.step_ms);
    let (mut power, mut bound) = (IDENTITY_SPRING_MATRIX, 1f64);
    for _ in 1..=config.max_steps_per_frame {
        power = multiply_spring_matrices(power, step);
        for entry in power {
            bound = bound.max(entry.abs());
        }
    }
    let declared = (bound * (1.0 + 1e-6)) as f32 as f64;
    assert!(declared.is_finite(), "the spring matrix overflows FP32 within max_steps_per_frame steps");
    declared
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SpringClock {
    /// How far springs have stepped, or `None` while the clock is stopped because nothing animates.
    pub animated_until_ms: Option<f64>,
    /// Settle checks run at check_origin_ms + k × settle_check_steps × step_ms.
    pub check_origin_ms: f64,
    /// The time of the last evaluation that wrote state back, or `None` after idle.
    pub last_evaluation_ms: Option<f64>,
}

pub const STOPPED_SPRING_CLOCK: SpringClock =
    SpringClock { animated_until_ms: None, check_origin_ms: 0.0, last_evaluation_ms: None };

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EvaluationKind {
    /// Runs steps up to a fixed simulated time and checks settling there, with no events.
    SettleCheck,
    /// Runs steps up to the events' timestamp, then applies every event with that timestamp.
    Events,
    /// Runs steps up to the frame's timestamp, for what's on screen.
    Frame,
    /// Runs no steps and snaps every animated object onto its destination with zero velocity.
    Snap,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WriteBack {
    /// Every evaluation writes next state back.
    EveryEvaluation,
    /// Only events, settle checks and snaps write back; a frame evaluates a copy, so stored state doesn't depend
    /// on which frames ran.
    EventsAndSettleChecks,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlannedEvaluation<Event> {
    pub kind: EvaluationKind,
    /// On the time grid.
    pub time_ms: f64,
    /// How far springs have stepped after this evaluation.
    pub simulated_time_ms: f64,
    /// Time since the last evaluation that wrote state back, or 0 after idle. Not clamped.
    pub duration_ms: f64,
    /// Write `spring_matrix_power(spring_step_matrix(constants, step_ms), steps)` into each matrix's 4 cells.
    pub steps: u32,
    /// Write 1 into the settle check cell when true.
    pub settle_check: bool,
    /// Write 1 into the snap cells when true; true only for a snap.
    pub snap: bool,
    /// The events this evaluation applies, all at `time_ms`, in delivery order.
    pub events: Vec<Event>,
    /// False for a frame with [`WriteBack::EventsAndSettleChecks`]: evaluate a copy of the table.
    pub writes_back: bool,
}

/// Plans and runs one frame's evaluations. `evaluate` writes the evaluation's matrices, settle check bit, snap bit
/// and events, runs the program, writes next state back when `writes_back` is true, and returns whether anything
/// still animates. Returns the clock after the frame; the host keeps requesting frames while
/// `animated_until_ms` is `Some`. `time_stamp` gives each event's timestamp.
///
/// Before an event with timestamp T applies, springs step to T toward the destinations in effect before T, so a
/// new destination starts pulling within one step of T at any frame rate. Settling is checked only at fixed
/// simulated times (and by an evaluation while the clock is stopped). A frame's evaluations run at most
/// `max_steps_per_frame` steps together; past that, a snap evaluation stops the clock and drops the backlog.
pub fn advance_frame<Event: Clone>(
    clock: SpringClock,
    frame_time_stamp: f64,
    events: &[Event],
    time_stamp: impl Fn(&Event) -> f64,
    mut evaluate: impl FnMut(&PlannedEvaluation<Event>) -> bool,
    write_back: WriteBack,
    config: &SpringClockConfig,
) -> SpringClock {
    validate_spring_clock_config(config);
    let check_interval_ms = config.step_ms * config.settle_check_steps as f64;
    let earliest_ms = clock.last_evaluation_ms.unwrap_or(f64::NEG_INFINITY);
    let frame_time_ms = quantize_time_ms(frame_time_stamp, config).max(earliest_ms);

    struct State {
        animated_until_ms: Option<f64>,
        check_origin_ms: f64,
        last_evaluation_ms: Option<f64>,
        step_budget: u32,
    }
    let mut state = State {
        animated_until_ms: clock.animated_until_ms,
        check_origin_ms: clock.check_origin_ms,
        last_evaluation_ms: clock.last_evaluation_ms,
        step_budget: config.max_steps_per_frame,
    };

    let mut run =
        |state: &mut State, kind: EvaluationKind, time_ms: f64, steps: u32, settle_check: bool, here: Vec<Event>| {
            let writes_back = kind != EvaluationKind::Frame || write_back == WriteBack::EveryEvaluation;
            let simulated_time_ms =
                state.animated_until_ms.map_or(time_ms, |until| until + steps as f64 * config.step_ms);
            let animating = evaluate(&PlannedEvaluation {
                kind,
                time_ms,
                simulated_time_ms,
                duration_ms: state.last_evaluation_ms.map_or(0.0, |last| time_ms - last),
                steps,
                settle_check,
                snap: kind == EvaluationKind::Snap,
                events: here,
                writes_back,
            });
            // A frame on a copy leaves the budget and the clock unchanged.
            if !writes_back {
                return;
            }
            state.step_budget -= steps;
            if state.animated_until_ms.is_none() && animating {
                // A stopped clock counts no time: an evaluation that finds something animating starts it.
                state.check_origin_ms = time_ms;
            }
            state.animated_until_ms = if animating { Some(simulated_time_ms) } else { None };
            state.last_evaluation_ms = if animating { Some(time_ms) } else { None };
        };

    let mut step_to = |state: &mut State, target_ms: f64, kind: EvaluationKind, here: Vec<Event>| {
        if let Some(until) = state.animated_until_ms
            && count_spring_steps(target_ms - until, state.step_budget, config).snap
        {
            // Past the budget: stop the clock, which drops the backlog, and snap.
            state.animated_until_ms = None;
            run(state, EvaluationKind::Snap, target_ms, 0, true, Vec::new());
        }
        // Settle checks at every fixed simulated time up to the target, including one where the target ends.
        while let Some(until) = state.animated_until_ms {
            let reachable_ms =
                until + count_spring_steps(target_ms - until, state.step_budget, config).steps as f64 * config.step_ms;
            let next_check_ms = state.check_origin_ms
                + (((until - state.check_origin_ms) / check_interval_ms).floor() + 1.0) * check_interval_ms;
            if next_check_ms > reachable_ms {
                break;
            }
            let time_ms = next_check_ms.max(state.last_evaluation_ms.unwrap_or(next_check_ms));
            let steps = ((next_check_ms - until) / config.step_ms) as u32;
            run(state, EvaluationKind::SettleCheck, time_ms, steps, true, Vec::new());
        }
        match state.animated_until_ms {
            None => run(state, kind, target_ms, 0, true, here),
            Some(until) => {
                let steps = count_spring_steps(target_ms - until, state.step_budget, config).steps;
                run(state, kind, target_ms, steps, false, here);
            }
        }
    };

    let mut timed: Vec<(f64, usize, &Event)> = events
        .iter()
        .enumerate()
        .map(|(index, event)| {
            let time_ms = frame_time_ms.min(earliest_ms.max(quantize_time_ms(time_stamp(event), config)));
            (time_ms, index, event)
        })
        .collect();
    timed.sort_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)));
    let mut start = 0;
    while start < timed.len() {
        let time_ms = timed[start].0;
        let mut end = start;
        while end < timed.len() && timed[end].0 == time_ms {
            end += 1;
        }
        let here: Vec<Event> = timed[start..end].iter().map(|&(_, _, event)| event.clone()).collect();
        step_to(&mut state, time_ms, EvaluationKind::Events, here);
        start = end;
    }
    step_to(&mut state, frame_time_ms, EvaluationKind::Frame, Vec::new());
    SpringClock {
        animated_until_ms: state.animated_until_ms,
        check_origin_ms: state.check_origin_ms,
        last_evaluation_ms: state.last_evaluation_ms,
    }
}
