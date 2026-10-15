//! A small UI panel, a toggle and a slider, whose every frame is computed by
//! a circuit compiled from one ordinary function: pointer handling, the
//! toggle's arming and flipping, slider dragging with clamping and snapping,
//! and the toggle's easing animation. Each frame's state is checked against
//! the function, then printed as CSV, one line per frame.
//!
//! cargo run --release --example ui_panel > frames.csv

use handwired_rs::TensorExecutor;
use handwired_rs::dsl::Params;
use handwired_rs::handwired;

// The panel is 480 × 200 px. The toggle's track spans x 40..144, y 40..96, and
// the slider's track runs from x 40 to 440 at y 150.
const TOGGLE: (i16, i16, i16, i16) = (40, 40, 144, 96);
const SLIDER_LEFT: i16 = 40;

/// One frame: the pointer's position and whether its button went down or up
/// this frame, applied to the panel's state. Returns the next state:
/// (on, armed, dragging, level in 0..=100, the toggle thumb's position 0..1).
#[handwired]
#[allow(clippy::too_many_arguments)]
fn ui_frame(
    on: bool,
    armed: bool,
    dragging: bool,
    #[range(0, 100)] level: u8,
    #[range(0, 1)] thumb: f32,
    #[range(0, 480)] x: i16,
    #[range(0, 200)] y: i16,
    pressed: bool,
    released: bool,
) -> (bool, bool, bool, u8, f32) {
    let mut on = on;
    let mut armed = armed;
    let mut dragging = dragging;
    let mut level = level;
    let mut thumb = thumb;
    let on_toggle = x >= 40 && x < 144 && y >= 40 && y < 96;
    let on_slider = x >= 28 && x < 452 && y >= 134 && y < 166;

    if pressed {
        armed = on_toggle;
        if on_slider {
            dragging = true;
        }
    }
    if dragging {
        // 400 px of track for 0..=100, snapped to steps of 5.
        let raw = ((x - 40) / 4).clamp(0, 100);
        level = ((raw + 2) / 5 * 5) as u8;
    }
    if released {
        if armed && on_toggle {
            on = !on;
        }
        armed = false;
        dragging = false;
    }

    // The thumb eases a third of the way to its side each frame, and snaps when close.
    let target: f32 = if on { 1.0 } else { 0.0 };
    thumb = (thumb + (target - thumb) * 0.3).clamp(0.0, 1.0);
    if (target - thumb).abs() < 0.01 {
        thumb = target;
    }
    (on, armed, dragging, level, thumb)
}

/// A scripted pointer: straight moves at a steady speed, presses and releases.
fn script() -> Vec<(i16, i16, bool, bool, bool)> {
    let mut frames = Vec::new();
    let mut at = (300i16, 20i16);
    let mut held = false;
    let move_to = |frames: &mut Vec<_>, at: &mut (i16, i16), held: bool, target: (i16, i16), steps: i16| {
        let start = *at;
        for step in 1..=steps {
            let x = start.0 + (target.0 - start.0) * step / steps;
            let y = start.1 + (target.1 - start.1) * step / steps;
            frames.push((x, y, false, false, held));
        }
        *at = target;
    };
    let pause = |frames: &mut Vec<_>, at: (i16, i16), held: bool, count: usize| {
        for _ in 0..count {
            frames.push((at.0, at.1, false, false, held));
        }
    };
    let press = |frames: &mut Vec<_>, at: (i16, i16), held: &mut bool| {
        *held = true;
        frames.push((at.0, at.1, true, false, true));
    };
    let release = |frames: &mut Vec<_>, at: (i16, i16), held: &mut bool| {
        *held = false;
        frames.push((at.0, at.1, false, true, false));
    };
    pause(&mut frames, at, held, 8);
    // Click the toggle on.
    move_to(&mut frames, &mut at, held, (92, 68), 14);
    press(&mut frames, at, &mut held);
    pause(&mut frames, at, held, 4);
    release(&mut frames, at, &mut held);
    pause(&mut frames, at, held, 14);
    // Drag the slider from 0 up to 75.
    move_to(&mut frames, &mut at, held, (SLIDER_LEFT, 150), 12);
    press(&mut frames, at, &mut held);
    move_to(&mut frames, &mut at, held, (346, 152), 26);
    pause(&mut frames, at, held, 4);
    release(&mut frames, at, &mut held);
    pause(&mut frames, at, held, 8);
    // Press the toggle, slide off it, and release: armed, then nothing happens.
    move_to(&mut frames, &mut at, held, (100, 70), 14);
    press(&mut frames, at, &mut held);
    pause(&mut frames, at, held, 4);
    move_to(&mut frames, &mut at, held, (220, 80), 8);
    release(&mut frames, at, &mut held);
    pause(&mut frames, at, held, 8);
    // Drag the slider back to 20, then click the toggle off.
    move_to(&mut frames, &mut at, held, (340, 150), 12);
    press(&mut frames, at, &mut held);
    move_to(&mut frames, &mut at, held, (120, 148), 20);
    release(&mut frames, at, &mut held);
    pause(&mut frames, at, held, 6);
    move_to(&mut frames, &mut at, held, (70, 66), 14);
    press(&mut frames, at, &mut held);
    pause(&mut frames, at, held, 3);
    release(&mut frames, at, &mut held);
    pause(&mut frames, at, held, 20);
    frames
}

fn main() {
    let compiled = ui_frame_circuit();
    let report = compiled.verify(20_000).unwrap();
    eprintln!(
        "ui_frame: {} live operations in {} layers; verified on {} sampled inputs",
        compiled.program.live_operation_count(),
        compiled.program.layers.len(),
        report.checked
    );
    let _ = (TOGGLE, SLIDER_LEFT);
    let mut executor = TensorExecutor::new(&compiled.program);
    let mut source = Vec::new();
    let mut state = (false, false, false, 0u8, 0.0f32);
    println!("frame,x,y,held,pressed,released,on,armed,dragging,level,thumb");
    for (frame, (x, y, pressed, released, held)) in script().into_iter().enumerate() {
        let args = (state.0, state.1, state.2, state.3, state.4, x, y, pressed, released);
        source.clear();
        args.encode(&mut source);
        let outputs = executor.run(&source);
        let next = (outputs[0] == 1.0, outputs[1] == 1.0, outputs[2] == 1.0, outputs[3] as u8, outputs[4]);
        let expected = ui_frame(state.0, state.1, state.2, state.3, state.4, x, y, pressed, released);
        assert_eq!(next, expected, "frame {frame}: the circuit disagrees with ui_frame()");
        state = next;
        println!(
            "{frame},{x},{y},{},{},{},{},{},{},{},{}",
            held as u8, pressed as u8, released as u8, state.0 as u8, state.1 as u8, state.2 as u8, state.3, state.4
        );
    }
}
