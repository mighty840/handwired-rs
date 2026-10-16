//! `#[handwired]` on scalar code: every circuit is verified against the
//! original function it came from.

// The edge coordinates are exact f32 values, written out in full.
#![allow(clippy::excessive_precision)]

use handwired_rs::handwired;

/// handwired's README example: `(a > 4) ? 10 : 12`.
#[handwired]
fn pick(a: u8) -> u8 {
    if a > 4 { 10 } else { 12 }
}

#[test]
fn readme_example() {
    let compiled = pick_circuit();
    for a in 0..=255u8 {
        assert_eq!(compiled.eval(&(a,)).unwrap(), pick(a));
    }
    // An integer comparison against a constant is 2 operations; the select is 4.
    assert!(compiled.program.live_operation_count() <= 8, "{}", compiled.program.live_operation_count());
}

/// handwired's template toggle switch, as an ordinary function: one
/// evaluation of the switch's logic, returning the next (on, armed).
#[handwired]
#[allow(clippy::too_many_arguments)]
fn toggle_step(
    on: bool,
    armed: bool,
    #[range(0, 16384)] viewport_width: i32,
    #[range(0, 16384)] viewport_height: i32,
    down: bool,
    #[range(-16384, 16384)]
    #[grid(1.0 / 64.0)]
    down_x: f32,
    #[range(-16384, 16384)]
    #[grid(1.0 / 64.0)]
    down_y: f32,
    up: bool,
    #[range(-16384, 16384)]
    #[grid(1.0 / 64.0)]
    up_x: f32,
    #[range(-16384, 16384)]
    #[grid(1.0 / 64.0)]
    up_y: f32,
    cancel: bool,
) -> (bool, bool) {
    let track_left = viewport_width as f32 * 0.5 - 26.0;
    let track_top = viewport_height as f32 * 0.5 - 16.0;
    let down_inside =
        down_x >= track_left && down_x < track_left + 52.0 && down_y >= track_top && down_y < track_top + 32.0;
    let up_inside = up_x >= track_left && up_x < track_left + 52.0 && up_y >= track_top && up_y < track_top + 32.0;
    let mut on = on;
    let mut armed = armed;
    if down {
        armed = down_inside;
    }
    if up && armed && up_inside {
        on = !on;
    }
    if up || cancel {
        armed = false;
    }
    (on, armed)
}

#[test]
fn toggle_matches_its_function() {
    let compiled = toggle_step_circuit();
    let report = compiled.verify(20_000).unwrap();
    assert_eq!(report.skipped, 0);
    // Presses near the track's edges, where an off-by-one would show.
    for x in [613.984_375f32, 614.0, 640.0, 665.984_375, 666.0] {
        for y in [383.984_375f32, 384.0, 400.0, 415.984_375, 416.0] {
            for (on, armed, down, up) in
                [(false, false, true, false), (false, true, false, true), (true, true, true, true)]
            {
                let args = (on, armed, 1280, 800, down, x, y, up, x, y, false);
                assert_eq!(compiled.eval(&args).unwrap(), compiled.oracle(args), "{args:?}");
            }
        }
    }
}

#[handwired]
fn arithmetic(a: u8, b: u8, #[range(-1000, 1000)] c: i16) -> (u16, i16, bool) {
    let sum = a as u16 + b as u16 * 3;
    let difference = c - a as i16;
    let either = a == b || c != 0;
    (sum, difference.abs(), either)
}

#[test]
fn integer_arithmetic_and_casts() {
    let report = arithmetic_circuit().verify(20_000).unwrap();
    assert_eq!(report.skipped, 0);
}

#[test]
fn batches_run_typed_records() {
    let compiled = arithmetic_circuit();
    let records: Vec<(u8, u8, i16)> =
        (0..1_000).map(|n: i32| ((n * 7 % 256) as u8, (n * 13 % 256) as u8, (n * 37 % 2001 - 1000) as i16)).collect();
    let results = compiled.run_batch(&records).unwrap();
    for (record, result) in records.iter().zip(results) {
        assert_eq!(result, arithmetic(record.0, record.1, record.2), "{record:?}");
    }
    assert!(matches!(compiled.run_batch(&[(0, 0, 2000)]), Err(handwired_rs::Error::SourceOutOfRange { .. })));
}

/// Overflow panics in the function, so those inputs don't constrain the circuit.
#[handwired]
fn overflowing(a: u8, b: u8) -> u8 {
    if a > b { a - b } else { a + b }
}

// Without overflow checks (a release build) the function wraps instead, and verify reports the mismatch.
#[cfg(debug_assertions)]
#[test]
fn panicking_inputs_are_skipped() {
    let report = overflowing_circuit().verify(5_000).unwrap();
    assert!(report.checked > 0 && report.skipped > 0, "{report:?}");
}

#[test]
fn verify_with_takes_inputs_from_a_sampler() {
    use handwired_rs::dsl::VerifyError;
    let compiled = arithmetic_circuit();
    // Only states a simulation could reach, say: a == b and small c.
    let mut sampled = 0;
    let report = compiled
        .verify_with(500, 1, |rng| {
            sampled += 1;
            let a = (rng.next_u64() % 256) as u8;
            (a, a, (rng.next_u64() % 21) as i16 - 10)
        })
        .unwrap();
    assert_eq!((report.checked, sampled), (500, 500));
    // A sampler can produce inputs outside the declared ranges, which is reported, not run.
    let error = compiled.verify_with(10, 1, |_| (0, 0, 5000)).unwrap_err();
    assert!(matches!(&error, VerifyError::OutOfRange { inputs, .. } if inputs == "(0, 0, 5000)"), "{error}");
}

const STAT_RANGES: [(f64, f64); 3] = [(2.0, 20.0), (1.0, 200.0), (2.0, 6.0)];

/// Damage, health and the roll to hit, each with its own range: from a list, or from a constant.
#[handwired]
fn expected_hits(
    #[ranges([(2, 20), (1, 200), (2, 6)])] stats: [f32; 3],
    #[ranges(STAT_RANGES)] other: [f32; 3],
) -> f32 {
    let hit_chance = (7.0 - stats[2]) / 8.0;
    stats[0] * hit_chance - other[1] / 64.0
}

#[test]
fn ranges_declare_one_range_per_scalar() {
    let compiled = expected_hits_circuit();
    for (cell, &(low, high)) in STAT_RANGES.iter().chain(&STAT_RANGES).enumerate() {
        let range = compiled.circuit.range(compiled.circuit.source(cell));
        assert_eq!((range.min, range.max), (low, high), "cell {cell}");
    }
    assert_eq!(compiled.verify(2_000).unwrap().skipped, 0);
    // Sampling stays within each scalar's own range.
    let report = compiled.verify_with(500, 3, |rng| {
        let stat = |rng: &mut handwired_rs::dsl::Rng, (low, high): (f64, f64)| (low + rng.unit() * (high - low)) as f32;
        (STAT_RANGES.map(|range| stat(rng, range)), STAT_RANGES.map(|range| stat(rng, range)))
    });
    assert_eq!(report.unwrap().checked, 500);
}

#[test]
fn encoded_records_run_and_differentiate_like_the_records() {
    use handwired_rs::BackpropOptions;
    let compiled = expected_hits_circuit();
    let records: Vec<([f32; 3], [f32; 3])> = (0..5_000)
        .map(|n| {
            let stats = [2.0 + (n % 18) as f32, 1.0 + (n % 199) as f32, 2.0 + (n % 5) as f32];
            (stats, stats)
        })
        .collect();
    let encoded = compiled.encode_records(&records).unwrap();
    assert_eq!((encoded.len(), encoded.sources().len()), (records.len(), records.len() * 6));
    let direct = compiled.run_batch(&records).unwrap();
    assert_eq!(compiled.run_encoded(&encoded), direct);
    for (record, result) in records.iter().zip(&direct) {
        assert_eq!(*result, expected_hits(record.0, record.1));
    }
    let weights: Vec<f32> = (0..records.len()).map(|n| [1.0, -0.5, 0.25][n % 3]).collect();
    let options = BackpropOptions::default();
    assert_eq!(
        compiled.gradients_encoded(&encoded, 0, Some(&weights), options).unwrap(),
        compiled.gradients(&records, 0, Some(&weights), options).unwrap()
    );
    // One record out of range fails the whole encoding.
    let mut bad = records.clone();
    bad[4_321].0[1] = 500.0;
    assert!(matches!(compiled.encode_records(&bad), Err(handwired_rs::Error::SourceOutOfRange { .. })));
}

/// A tuning-shaped function: shared parameters, then a per-record roll.
#[handwired]
fn tuned_hit(#[ranges([(1, 20), (1, 6)])] params: [f32; 2], #[range(1, 6)] roll: u8) -> f32 {
    if roll as f32 >= params[1] { params[0] } else { 0.0 }
}

#[test]
fn broadcast_rewrites_one_argument_of_every_encoded_record() {
    use handwired_rs::BackpropOptions;
    use handwired_rs::dsl::Params;
    let compiled = tuned_hit_circuit();
    assert_eq!((<([f32; 2], u8)>::arg_cells(0), <([f32; 2], u8)>::arg_cells(1)), (0..2, 2..3));
    let rolls: Vec<u8> = (0..10_000).map(|n| 1 + (n % 6) as u8).collect();
    let records = |params: [f32; 2]| -> Vec<([f32; 2], u8)> { rolls.iter().map(|&roll| (params, roll)).collect() };
    let mut encoded = compiled.encode_records(&records([5.0, 4.0])).unwrap();
    for params in [[7.5f32, 3.0], [2.0, 5.5], [19.0, 1.0]] {
        compiled.broadcast(&mut encoded, 0, &params).unwrap();
        let fresh = compiled.encode_records(&records(params)).unwrap();
        assert_eq!(encoded.sources(), fresh.sources());
        assert_eq!(compiled.run_encoded(&encoded), compiled.run_batch(&records(params)).unwrap());
        let options = BackpropOptions::default();
        assert_eq!(
            compiled.gradients_encoded(&encoded, 0, None, options).unwrap(),
            compiled.gradients(&records(params), 0, None, options).unwrap()
        );
    }
    // Out of the argument's ranges: refused, and the records keep their values.
    let before = encoded.sources().to_vec();
    assert!(matches!(
        compiled.broadcast(&mut encoded, 0, &[25.0f32, 3.0]),
        Err(handwired_rs::Error::SourceOutOfRange { .. })
    ));
    assert_eq!(encoded.sources(), &before[..]);
    // A value of the wrong size panics.
    let wrong = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        compiled.broadcast(&mut encoded, 0, &[1.0f32, 2.0, 3.0])
    }));
    assert!(wrong.is_err());
}
