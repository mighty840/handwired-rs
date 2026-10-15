//! A toggle switch centered in the viewport, as one numeric program, ported
//! from handwired's template/toggle.ts without its spring animation. Like a
//! native switch, it acts on release: a press on the track arms it, a release
//! on the track while armed flips it, and a release off the track or a cancel
//! disarms it. `ToggleReference` is the same switch in plain Rust, and
//! tests/toggle.rs compares the two.
//!
//! Run with `cargo run --example toggle`.
#![allow(dead_code)]

use handwired_rs::{emit_and, emit_compare, emit_not, emit_select, event, input, state_bit, Circuit, Relation, SourceTable, TensorProgram, Wire};

pub const TRACK_WIDTH: f64 = 52.0;
pub const TRACK_HEIGHT: f64 = 32.0;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum PointerKind {
    Down,
    Up,
    Cancel,
}

/// A pointer event, with coordinates on the 1/64 px grid.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PointerEvent {
    pub kind: PointerKind,
    pub x: f32,
    pub y: f32,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Output {
    pub track_left: f32,
    pub track_top: f32,
    pub on: f32,
    pub armed: f32,
}

struct Pointer {
    active: Wire,
    x: Wire,
    y: Wire,
}

pub struct Toggle {
    table: SourceTable,
    circuit: Circuit,
    program: TensorProgram,
    source: Vec<f32>,
    viewport: (Wire, Wire),
    down: Pointer,
    up: Pointer,
    cancel: Wire,
    outputs: [Wire; 4],
}

impl Default for Toggle {
    fn default() -> Self {
        Self::new()
    }
}

impl Toggle {
    pub fn new() -> Toggle {
        // Every number the program reads, or keeps from one evaluation to the next.
        let mut table = SourceTable::new();
        let coordinate = || input(-16_384.0, 16_384.0).grid(1.0 / 64.0);
        let viewport_width = table.declare("viewport.width", input(0.0, 16_384.0).grid(1.0).initial(1280.0));
        let viewport_height = table.declare("viewport.height", input(0.0, 16_384.0).grid(1.0).initial(800.0));
        let mut pointer = |name: &str| Pointer {
            active: table.declare(format!("{name}.active"), event()),
            x: table.declare(format!("{name}.x"), coordinate()),
            y: table.declare(format!("{name}.y"), coordinate()),
        };
        let down = pointer("down");
        let up = pointer("up");
        let cancel = table.declare("cancel", event());
        let on_cell = table.declare("on", state_bit());
        let armed_cell = table.declare("armed", state_bit());
        let mut circuit = table.create_circuit();
        let c = &mut circuit;

        // Layout. trackLeft = viewportWidth / 2 - trackWidth / 2
        let track_left = c.affine(viewport_width * 0.5 - TRACK_WIDTH / 2.0);
        let track_top = c.affine(viewport_height * 0.5 - TRACK_HEIGHT / 2.0);
        let track_right = c.affine(track_left + TRACK_WIDTH);
        let track_bottom = c.affine(track_top + TRACK_HEIGHT);
        // x >= trackLeft && x < trackLeft + trackWidth && y >= trackTop && y < trackTop + trackHeight
        let inside_track = |c: &mut Circuit, point: &Pointer| {
            let bits = [
                emit_compare(c, point.x, Relation::AtLeast, track_left),
                emit_compare(c, point.x, Relation::Less, track_right),
                emit_compare(c, point.y, Relation::AtLeast, track_top),
                emit_compare(c, point.y, Relation::Less, track_bottom),
            ];
            emit_and(c, &bits)
        };

        // Handle inputs in a fixed order: down, then up, then cancel.
        // if (down) armed = insideTrack(down)
        let down_inside = inside_track(c, &down);
        let armed_after_down = emit_select(c, down.active, armed_cell, down_inside);
        // if (up && armed && insideTrack(up)) on = !on
        let up_inside = inside_track(c, &up);
        let flips = emit_and(c, &[up.active, armed_after_down, up_inside]);
        let flipped = emit_not(c, on_cell);
        let on = emit_select(c, flips, on_cell, flipped);
        // if (up || cancel) armed = false
        let no_up = emit_not(c, up.active);
        let no_cancel = emit_not(c, cancel);
        let armed = emit_and(c, &[armed_after_down, no_up, no_cancel]);

        // Commit state: each state cell gets exactly one next value.
        table.next(on_cell, on);
        table.next(armed_cell, armed);
        table.finish();
        let outputs = [track_left, track_top, on, armed];
        let mut roots = outputs.to_vec();
        roots.extend(table.next_wires());
        let program = circuit.compile(Some(&roots));
        let source = table.initial();
        Toggle {
            table,
            circuit,
            program,
            source,
            viewport: (viewport_width, viewport_height),
            down,
            up,
            cancel,
            outputs,
        }
    }

    pub fn circuit(&self) -> &Circuit {
        &self.circuit
    }

    pub fn program(&self) -> &TensorProgram {
        &self.program
    }

    /// One evaluation: write the viewport and this frame's events into the
    /// table, run the compiled program, and keep the next state.
    pub fn step(&mut self, width: f32, height: f32, events: &[PointerEvent]) -> Result<Output, String> {
        let mut values = self.source.clone();
        values[self.viewport.0.index()] = width;
        values[self.viewport.1.index()] = height;
        for event in events {
            let pointer = match event.kind {
                PointerKind::Down => &self.down,
                PointerKind::Up => &self.up,
                PointerKind::Cancel => {
                    values[self.cancel.index()] = 1.0;
                    continue;
                }
            };
            values[pointer.active.index()] = 1.0;
            values[pointer.x.index()] = event.x;
            values[pointer.y.index()] = event.y;
        }
        self.table.validate(&values)?;
        let result = self.circuit.evaluate_tensor(&values, &self.program).map_err(|error| error.to_string())?;
        self.source = self.table.advance(&values, &result);
        let [track_left, track_top, on, armed] = self.outputs.map(|wire| result[wire.index()]);
        Ok(Output { track_left, track_top, on, armed })
    }
}

/// The same switch in plain Rust, in f64.
#[derive(Default)]
pub struct ToggleReference {
    on: bool,
    armed: bool,
}

impl ToggleReference {
    pub fn step(&mut self, width: f64, height: f64, events: &[PointerEvent]) -> Output {
        let track_left = width / 2.0 - TRACK_WIDTH / 2.0;
        let track_top = height / 2.0 - TRACK_HEIGHT / 2.0;
        let inside = |event: &PointerEvent| {
            let (x, y) = (event.x as f64, event.y as f64);
            x >= track_left && x < track_left + TRACK_WIDTH && y >= track_top && y < track_top + TRACK_HEIGHT
        };
        let last = |kind| events.iter().rev().find(|event| event.kind == kind);
        if let Some(down) = last(PointerKind::Down) {
            self.armed = inside(down);
        }
        let up = last(PointerKind::Up);
        if let Some(up) = up
            && self.armed && inside(up) {
                self.on = !self.on;
            }
        if up.is_some() || last(PointerKind::Cancel).is_some() {
            self.armed = false;
        }
        let bit = |value: bool| if value { 1.0 } else { 0.0 };
        Output { track_left: track_left as f32, track_top: track_top as f32, on: bit(self.on), armed: bit(self.armed) }
    }
}

fn main() {
    let mut toggle = Toggle::new();
    let program = toggle.program();
    println!(
        "toggle: {} source cells, {} built operations, {} live in {} layers, {} stored weights",
        toggle.circuit().source_size(),
        toggle.circuit().operations().len(),
        program.live_operation_count(),
        program.layers.len(),
        program.stored_weight_count(),
    );
    let press = |kind, x, y| [PointerEvent { kind, x, y }];
    let frames: [(&str, Vec<PointerEvent>); 4] = [
        ("press on the track", press(PointerKind::Down, 640.0, 400.0).to_vec()),
        ("release on the track", press(PointerKind::Up, 640.0, 400.0).to_vec()),
        ("press 1/64 px left of the track", press(PointerKind::Down, 613.984_4, 400.0).to_vec()),
        ("release on the track", press(PointerKind::Up, 640.0, 400.0).to_vec()),
    ];
    for (label, events) in frames {
        let output = toggle.step(1280.0, 800.0, &events).unwrap();
        println!("{label:32} -> on {} armed {}", output.on, output.armed);
    }
}
