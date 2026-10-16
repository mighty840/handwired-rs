//! Runs the toggle program and its plain Rust version over scripted and
//! random traces, and compares them every frame. Ported from handwired's
//! template/toggle.test.ts, without the spring.

#![allow(clippy::type_complexity)]

mod common;
#[path = "../examples/toggle.rs"]
mod toggle;

use common::Rng;
use toggle::{PointerEvent, PointerKind, Toggle, ToggleReference};

fn pointer(kind: PointerKind, x: f32, y: f32) -> Vec<PointerEvent> {
    vec![PointerEvent { kind, x, y }]
}

#[test]
fn scripted_trace_matches_reference_and_each_step_takes_effect() {
    use PointerKind::*;
    // At 1280 × 800 the track spans x 614..666 and y 384..416. (events, expected on, expected armed)
    let trace: Vec<(Vec<PointerEvent>, f32, f32)> = vec![
        (vec![], 0.0, 0.0),
        (pointer(Down, 640.0, 400.0), 0.0, 1.0), // a press on the track arms the switch
        (pointer(Up, 640.0, 400.0), 1.0, 0.0),   // a release on the track while armed flips it on
        (pointer(Down, 630.25, 399.5), 1.0, 1.0),
        (pointer(Up, 630.25, 399.5), 0.0, 0.0),      // and back off
        (pointer(Down, 613.984_4, 400.0), 0.0, 0.0), // 1/64 px left of the track
        (pointer(Down, 614.0, 384.0), 0.0, 1.0),     // exactly on the left and top edges
        (pointer(Up, 666.0, 400.0), 0.0, 0.0),       // the right edge is outside, so it only disarms
        (pointer(Down, 640.0, 400.0), 0.0, 1.0),
        (pointer(Cancel, 0.0, 0.0), 0.0, 0.0), // a cancel disarms
        (pointer(Up, 640.0, 400.0), 0.0, 0.0), // so this release doesn't flip
        (pointer(Down, 665.984_4, 415.984_38), 0.0, 1.0),
        (pointer(Up, 665.984_4, 415.984_38), 1.0, 0.0), // 1/64 px inside the bottom right corner
    ];
    let mut program = Toggle::new();
    let mut reference = ToggleReference::default();
    for (events, on, armed) in trace {
        let output = program.step(1280.0, 800.0, &events).unwrap();
        assert_eq!(output, reference.step(1280.0, 800.0, &events));
        assert_eq!((output.on, output.armed), (on, armed), "after {events:?}");
    }
}

#[test]
fn events_in_one_evaluation_apply_down_then_up_then_cancel() {
    use PointerKind::*;
    let at = |kind| PointerEvent { kind, x: 640.0, y: 400.0 };
    let cases: Vec<(Vec<Vec<PointerEvent>>, (f32, f32))> = vec![
        // A press, then a release and a press together: the press applies first, and the release flips and disarms.
        (vec![vec![at(Down)], vec![at(Up), at(Down)]], (1.0, 0.0)),
        // A cancel and a press together: the cancel applies after the press.
        (vec![vec![at(Cancel), at(Down)]], (0.0, 0.0)),
    ];
    for (frames, expected) in cases {
        let mut program = Toggle::new();
        let mut reference = ToggleReference::default();
        let mut decisions = (0.0, 0.0);
        for events in &frames {
            let output = program.step(1280.0, 800.0, events).unwrap();
            assert_eq!(output, reference.step(1280.0, 800.0, events));
            decisions = (output.on, output.armed);
        }
        assert_eq!(decisions, expected);
    }
}

#[test]
fn random_traces_near_the_edges_match_reference() {
    let mut rng = Rng::new(2026);
    let kinds = [PointerKind::Down, PointerKind::Up, PointerKind::Cancel];
    for _ in 0..50 {
        let mut program = Toggle::new();
        let mut reference = ToggleReference::default();
        for _ in 0..200 {
            // Odd viewport sizes put the track on half pixels.
            let width = rng.pick(&[1280.0, 1281.0, 333.0, 52.0, 0.0]);
            let height = rng.pick(&[800.0, 801.0, 32.0, 17.0]);
            let near = |rng: &mut Rng, edge: f64| ((edge + rng.uniform(-2.0, 2.0)) * 64.0).round() / 64.0;
            let left = width / 2.0 - 26.0;
            let top = height / 2.0 - 16.0;
            let events: Vec<PointerEvent> = (0..rng.below(3))
                .map(|_| {
                    let kind = rng.pick(&kinds);
                    let (edge_x, edge_y) =
                        (rng.pick(&[left, left + 52.0, left + 26.0]), rng.pick(&[top, top + 32.0, top + 16.0]));
                    PointerEvent { kind, x: near(&mut rng, edge_x) as f32, y: near(&mut rng, edge_y) as f32 }
                })
                .collect();
            let output = program.step(width as f32, height as f32, &events).unwrap();
            assert_eq!(output, reference.step(width, height, &events), "{width}×{height} {events:?}");
        }
    }
}

#[test]
fn the_table_check_rejects_a_coordinate_off_the_grid() {
    let mut program = Toggle::new();
    let error = program.step(1280.0, 800.0, &pointer(PointerKind::Down, 640.001, 400.0)).unwrap_err();
    assert!(error.contains("down.x must be a multiple of 0.015625"), "{error}");
}
