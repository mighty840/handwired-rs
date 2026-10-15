//! `#[handwired]` on scalar code: every circuit is verified against the
//! original function it came from.

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

/// Overflow panics in the function, so those inputs don't constrain the circuit.
#[handwired]
fn overflowing(a: u8, b: u8) -> u8 {
    if a > b { a - b } else { a + b }
}

#[test]
fn panicking_inputs_are_skipped() {
    let report = overflowing_circuit().verify(5_000).unwrap();
    assert!(report.checked > 0 && report.skipped > 0, "{report:?}");
}
