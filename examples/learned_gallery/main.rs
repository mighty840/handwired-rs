//! A photo layout drawn by a small network, after handwired's learned gallery demo.
//!
//! The network's 244 weights are the program's source cells. Training steps them with Adam, using the circuit's
//! backward pass, until one rectangle per photo matches a sketch, while a second part of the loss pulls photo edges
//! that are nearly on one line onto one line. This runs the page's three preset sketches to rest, then nudges one
//! rectangle of a row within the penalty's reach, so its photo stays in the row, and farther, so it lets go. The
//! page's pointer editor and drawing are not ported; every frame trains 10 steps, as a 60 Hz frame of the page does.
//!
//! cargo run --release --example learned_gallery

mod gallery;

use gallery::{
    DEFAULT_HOLD_DISTANCE, Edge, Gallery, Preset, Rect, STEPS_PER_FRAME, largest_error, line_spread, sketch_lines,
};

// The stage that the px below are on, the page's at a common window size.
const STAGE_WIDTH: f64 = 538.0;
const STAGE_HEIGHT: f64 = 430.0;

fn px(fraction: f64, edge: Edge) -> f64 {
    fraction * if edge.across() { STAGE_WIDTH } else { STAGE_HEIGHT }
}

/// The widest line of photos, in px: the largest spread of the photo edges along one line of the sketch.
fn widest_line(gallery: &Gallery) -> f64 {
    let machine = &gallery.machine;
    sketch_lines(&machine.target)
        .iter()
        .map(|(edge, slots)| px(line_spread(machine, &gallery.last.values, slots, *edge), *edge))
        .fold(0.0, f64::max)
}

fn main() {
    let first = Gallery::preset(Preset::Story);
    let network = &first.machine.network;
    println!(
        "network: {} weights, {} operations in {} layers",
        network.parameter_count(),
        network.program.live_operation_count(),
        network.program.layers.len()
    );
    println!(
        "frames of {STEPS_PER_FRAME} steps, misalignment penalty at {}% of the stage's width",
        DEFAULT_HOLD_DISTANCE * 100.0
    );
    println!("px on a stage of {STAGE_WIDTH} by {STAGE_HEIGHT}\n");

    for preset in Preset::ALL {
        let mut gallery = Gallery::preset(preset);
        let undone = gallery.settle();
        let last = &gallery.last;
        println!(
            "{:<14} {:>2} photos: at rest after {:>4} steps ({undone} frames taken back), loss {:.1e}, \
             largest error {:.2}% of the stage, widest line {:.3} px",
            preset.name(),
            gallery.machine.target.active_count(),
            last.step,
            last.loss,
            largest_error(&gallery.machine, &last.values) * 100.0,
            widest_line(&gallery),
        );
    }

    // Contact sheet's second row is slots 4 to 7. A photo lets go of its line at PENALTY_STRENGTH reaches, i.e. at
    // the slider's distance: 0.04 of the stage's width, 0.05 of its height.
    let let_go = DEFAULT_HOLD_DISTANCE * 5.0 / 4.0;
    println!(
        "\nContact sheet, rectangle 5 dragged down; its photo lets go of the row at {:.1} px",
        px(let_go, Edge::Top)
    );
    for nudge in [0.7, 1.3] {
        let mut gallery = Gallery::preset(Preset::Contact);
        gallery.settle();
        let drawn = gallery.machine.target.rect(5);
        let moved = nudge * let_go;
        gallery.set_rect(5, Rect { top: drawn.top + moved, ..drawn });
        gallery.settle();
        let machine = &gallery.machine;
        let values = &gallery.last.values;
        let target_items = machine.active_target_items();
        let outputs = machine.match_rectangles(values, &target_items);
        let photo_top = |slot: usize| {
            let index = target_items.iter().position(|&item| item == slot).unwrap();
            machine.output_rect(values, outputs[index]).top
        };
        let others = line_spread(machine, values, &[4, 6, 7], Edge::Top);
        println!(
            "  by {:>4.1} px: photo 5's top is {:.2} px from the row's and {:.2} px from its rectangle's, \
             the other three tops within {:.3} px ({} steps)",
            px(moved, Edge::Top),
            px((photo_top(5) - photo_top(4)).abs(), Edge::Top),
            px((drawn.top + moved - photo_top(5)).abs(), Edge::Top),
            px(others, Edge::Top),
            gallery.last.step,
        );
    }
}
