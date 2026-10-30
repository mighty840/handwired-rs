//! The learned gallery example's network and training, after handwired's tests/learned-gallery: the program's size,
//! the penalty's pull against its cost, fits of the presets and of layouts that are no preset, frames that are taken
//! back, and photos that hold a line or let go of it.

#[path = "../examples/learned_gallery/gallery.rs"]
mod gallery;

use gallery::{
    DEFAULT_HOLD_DISTANCE, Edge, Gallery, MAXIMUM_TRAINING_STEPS_AFTER_CHANGE, Machine, PENALTY_STRENGTH, Preset, Rect,
    SLOTS, TRAINING_STEPS_AFTER_CHANGE, Target, largest_error, line_spread, pair_cost, pair_pull, sketch_lines,
};

#[test]
fn the_program_is_the_size_of_handwireds_and_its_tensor_form_equals_the_reference() {
    let mut machine = Machine::new(Target::new(&Preset::Story.rects()));
    let network = &machine.network;
    // handwired's README: 244 source cells, 6,320 operations in 8 layers. Every product multiplies two wires
    // without a declared sign, so it is 3 operations.
    assert_eq!(network.parameter_count(), 244);
    assert_eq!(network.program.live_operation_count(), 6320);
    assert_eq!(network.program.layers.len(), 8);
    assert_eq!(network.geometries.len(), SLOTS * 4);
    for _ in 0..2 {
        let network = &machine.network;
        let reference = network.circuit.evaluate_reference(&machine.source).unwrap();
        let tensor = network.circuit.evaluate_tensor(&machine.source, &network.program).unwrap();
        for wire in &network.geometries {
            assert_eq!(reference[wire.index()].to_bits(), tensor[wire.index()].to_bits());
        }
        machine.train(50);
    }
}

#[test]
fn the_penaltys_pull_is_the_slope_of_its_cost() {
    let reach = 0.02;
    for multiplier in [0.0, 0.004, -0.004, 0.019, -0.019] {
        for index in -60..=60 {
            let gap = index as f64 * 0.0005 + 0.0001;
            // The pull has corners, so the cost's slope over a small step differs from the pull by up to the step
            // times the pull's steepest change.
            let step = 1e-6;
            let slope =
                (pair_cost(gap + step, multiplier, reach) - pair_cost(gap - step, multiplier, reach)) / (2.0 * step);
            let pull = pair_pull(gap, multiplier, reach);
            assert!((slope - pull).abs() < 1e-5, "multiplier {multiplier}, gap {gap}: slope {slope}, pull {pull}");
        }
        // The mirrored pair is charged the same and pulled the other way.
        assert_eq!(pair_cost(-0.003, -multiplier, reach), pair_cost(0.003, multiplier, reach));
        assert_eq!(pair_pull(-0.003, -multiplier, reach), -pair_pull(0.003, multiplier, reach));
    }
    // Without a multiplier the pull is 5 times a small gap, and the reach minus the gap from where the two meet.
    assert!((pair_pull(0.001, 0.0, reach) - 0.005).abs() < 1e-12);
    assert!((pair_pull(0.015, 0.0, reach) - 0.005).abs() < 1e-12);
    assert!((pair_pull(-0.015, 0.0, reach) + 0.005).abs() < 1e-12);
    // A multiplier adds to the pull, within the same limit.
    assert!((pair_pull(0.001, 0.004, reach) - 0.009).abs() < 1e-12);
    assert!((pair_pull(0.001, 0.019, reach) - 0.019).abs() < 1e-12);
    // Edges a reach or more apart aren't pulled, and their cost stays what it is at the reach.
    for gap in [0.02, 0.03, -0.02, -0.5] {
        assert_eq!(pair_pull(gap, 0.0, reach).abs(), 0.0);
        assert_eq!(pair_pull(gap, 0.01, reach).abs(), 0.0);
        assert_eq!(pair_cost(gap, 0.0, reach), pair_cost(reach, 0.0, reach));
    }
    assert_eq!(pair_cost(0.0, 0.0, reach), 0.0);
    assert!((pair_cost(0.001, 0.0, reach) - 0.5 * 5.0 * 0.001f64.powi(2)).abs() < 1e-15);
    // With a reach of 0 the penalty is off.
    assert_eq!(pair_pull(0.001, 0.0, 0.0).abs(), 0.0);
    assert_eq!(pair_cost(0.001, 0.0, 0.0), 0.0);
}

fn sidebar_grid() -> Vec<Rect> {
    let gap = 0.025;
    let sidebar_width = 0.3;
    let cell_width = (1.0 - sidebar_width - gap * 4.0) / 3.0;
    let cell_height = (1.0 - gap * 5.0) / 4.0;
    let mut rects = vec![Rect { left: gap, top: gap, width: sidebar_width, height: 1.0 - gap * 2.0 }];
    for item in 0..11 {
        rects.push(Rect {
            left: sidebar_width + gap * 2.0 + (item % 3) as f64 * (cell_width + gap),
            top: gap + (item / 3) as f64 * (cell_height + gap),
            width: cell_width,
            height: cell_height,
        });
    }
    rects
}

/// The root mean square error of the fields of the photos against their rectangles.
fn matched_rms_error(machine: &Machine, values: &[f32]) -> f64 {
    let target_items = machine.active_target_items();
    let outputs = machine.match_rectangles(values, &target_items);
    let mut squared = 0.0;
    for (&target_item, &output) in target_items.iter().zip(&outputs) {
        let (actual, expected) = (machine.output_rect(values, output), machine.target.rect(target_item));
        for error in [
            actual.left - expected.left,
            actual.top - expected.top,
            actual.width - expected.width,
            actual.height - expected.height,
        ] {
            squared += error * error;
        }
    }
    (squared / (target_items.len() * 4) as f64).sqrt()
}

#[test]
fn the_shared_weights_fit_a_preset_and_layouts_that_are_no_preset_in_any_slots() {
    let mut cinema = Machine::new(Target::new(&Preset::Cinema.rects()));
    let initial = cinema.evaluate();
    let trained = cinema.train(500);
    assert!(trained.loss < 0.000_01 && trained.loss < initial.loss * 0.01, "Cinema loss {}", trained.loss);

    let masonry = [
        Rect { left: 0.025, top: 0.025, width: 0.3, height: 0.44 },
        Rect { left: 0.35, top: 0.025, width: 0.28, height: 0.28 },
        Rect { left: 0.655, top: 0.025, width: 0.32, height: 0.58 },
        Rect { left: 0.025, top: 0.49, width: 0.3, height: 0.485 },
        Rect { left: 0.35, top: 0.33, width: 0.28, height: 0.645 },
        Rect { left: 0.655, top: 0.63, width: 0.32, height: 0.345 },
    ];
    for (rects, ceiling) in [(sidebar_grid(), 0.005), (masonry.to_vec(), 0.007)] {
        let mut machine = Machine::new(Target::new(&rects));
        let trained = machine.train(500);
        assert!(trained.loss < 0.000_01, "loss {}", trained.loss);
        assert!(matched_rms_error(&machine, &trained.values) < ceiling);
        assert!(machine.source.iter().zip(&machine.initial_source).any(|(a, b)| (a - b).abs() > 0.01));
    }

    // The sidebar grid reversed, in scattered slots, reaches the same fit: the slots don't steer the layout.
    let slots = [0, 2, 5, 7, 9, 12, 13, 14, 16, 17, 18, 19];
    let mut target = Target::new(&[]);
    for (&slot, &rect) in slots.iter().zip(sidebar_grid().iter().rev()) {
        target.set(slot, rect);
    }
    let mut permuted = Machine::new(target);
    let trained = permuted.train(500);
    assert!(trained.loss < 0.000_01, "permuted loss {}", trained.loss);
    assert!(matched_rms_error(&permuted, &trained.values) < 0.005);
}

#[test]
fn reset_restores_the_seeded_weights() {
    let mut machine = Machine::new(Target::new(&Preset::Story.rects()));
    let seeded = machine.source.clone();
    machine.train(40);
    assert_eq!(machine.training_step(), 40);
    assert_ne!(machine.source, seeded);
    machine.reset();
    assert_eq!(machine.training_step(), 0);
    assert_eq!(machine.source, seeded);
}

#[test]
fn steps_that_raise_the_loss_are_taken_back() {
    // Adam at a fixed learning rate doesn't come to rest: after Cinema's first 500 steps, some frames of 10 more
    // plain steps raise the loss many times over. The penalty is off, so the loss is the distance alone.
    let mut plain = Machine::new(Target::new(&Preset::Cinema.rects()));
    let mut loss = plain.train(TRAINING_STEPS_AFTER_CHANGE).loss;
    let after_first_steps = loss;
    let mut largest_rise: f64 = 1.0;
    for _ in 0..150 {
        let next = plain.train(10).loss;
        largest_rise = largest_rise.max(next / loss);
        loss = next;
    }
    assert!(largest_rise > 20.0, "the largest rise was {largest_rise} times");

    // train_or_undo takes such a frame's steps back: the loss never rises, and the weights and the outputs are the
    // ones from before the frame.
    let mut guarded = Machine::new(Target::new(&Preset::Cinema.rects()));
    loss = guarded.train(TRAINING_STEPS_AFTER_CHANGE).loss;
    let mut undone = 0;
    for frame in 0..150 {
        let source_before = guarded.source.clone();
        let values_before = guarded.evaluate().values;
        let outcome = guarded.train_or_undo(10);
        assert!(outcome.loss <= loss, "frame {frame} raised the loss from {loss} to {}", outcome.loss);
        if outcome.undone {
            undone += 1;
            assert_eq!(outcome.loss, loss);
            assert_eq!(guarded.source, source_before);
            assert_eq!(outcome.values, values_before);
        }
        loss = outcome.loss;
    }
    assert!((5..=100).contains(&undone), "{undone} frames were undone");
    // The steps that were taken back still count.
    assert_eq!(guarded.training_step(), TRAINING_STEPS_AFTER_CHANGE + 1500);
    assert!(loss < after_first_steps * 0.1, "{loss} against {after_first_steps} after the first steps");
}

#[test]
fn the_penalty_pulls_edges_within_its_reach_onto_one_line_and_leaves_the_others() {
    // Three rectangles side by side. The tops of the first two are 0.012 apart, and the third's is 0.05 below the
    // first's. The bottoms are 0.2 and more apart.
    let drawn = [
        Rect { left: 0.05, top: 0.1, width: 0.25, height: 0.2 },
        Rect { left: 0.375, top: 0.112, width: 0.25, height: 0.4 },
        Rect { left: 0.7, top: 0.15, width: 0.25, height: 0.75 },
    ];
    let fit = |reach: f64| {
        let mut machine = Machine::new(Target::new(&drawn));
        machine.set_penalty_reach(reach, reach);
        let mut outcome = machine.train_or_undo(10);
        for _ in 1..200 {
            outcome = machine.train_or_undo(10);
        }
        let outputs = machine.match_rectangles(&outcome.values, &[0, 1, 2]);
        let rects: Vec<Rect> = outputs.iter().map(|&output| machine.output_rect(&outcome.values, output)).collect();
        (outcome, rects)
    };
    // Without the penalty the photos are where the sketch has them.
    let (plain, rects) = fit(0.0);
    assert_eq!(plain.misalignment, 0.0);
    assert!(plain.loss < 1e-8, "loss {}", plain.loss);
    assert!((rects[1].top - rects[0].top - 0.012).abs() < 0.0005);
    // With a reach of 0.02 the first two tops are on one line, at the mean of where they are drawn, and the third
    // is beyond the reach and stays.
    let (pulled, rects) = fit(0.02);
    assert!((rects[0].top - rects[1].top).abs() < 0.0001, "the tops are {} apart", (rects[0].top - rects[1].top).abs());
    assert!((rects[0].top - 0.106).abs() < 0.0005, "the line is at {}", rects[0].top);
    assert!((rects[2].top - 0.15).abs() < 0.0005, "the third top is at {}", rects[2].top);
    for (rect, drawn) in rects.iter().zip(&drawn) {
        assert!((rect.left - drawn.left).abs() < 0.0005);
        assert!((rect.left + rect.width - drawn.left - drawn.width).abs() < 0.0005);
        assert!((rect.top + rect.height - drawn.top - drawn.height).abs() < 0.0005);
    }
    // The loss rests where its two parts balance: two held edges each 0.006 from where they are drawn, as a mean
    // over 12 edges, and a closed gap.
    assert!((pulled.from_target - 2.0 * 0.5 * 0.006f64.powi(2) / 12.0).abs() < 5e-7, "{}", pulled.from_target);
    assert!(pulled.misalignment < 1e-8, "penalty {}", pulled.misalignment);
    // An edge lets go at PENALTY_STRENGTH reaches: with a reach of 0.005 the two tops, 0.012 apart, stay as drawn.
    let (_, rects) = fit(0.005);
    assert!((rects[1].top - rects[0].top - 0.012).abs() < 0.0005);
}

#[test]
fn each_preset_is_fitted_closely_with_its_photos_in_line_and_training_stops_by_itself() {
    for (preset, count) in [(Preset::Story, 10), (Preset::Cinema, 9), (Preset::Contact, 11)] {
        let name = preset.name();
        let mut gallery = Gallery::preset(preset);
        assert_eq!(gallery.machine.target.active_count(), count);
        // The seeded weights put the photos in a pile, and the penalty is off while they leave it.
        assert!(gallery.last.loss > 0.005, "{name} starts at loss {}", gallery.last.loss);
        while gallery.last.step < 100 {
            assert!(gallery.frame());
        }
        assert_eq!(gallery.last.misalignment, 0.0);
        // It comes on once the photos are close to the sketch.
        while gallery.last.misalignment == 0.0 {
            assert!(gallery.frame());
        }
        let onset = gallery.last.step;
        assert!(onset > 100 && onset < 400, "{name}: the penalty came on at step {onset}");
        assert!(
            gallery.last.from_target < 2e-6,
            "{name}: the distance from the sketch was {}",
            gallery.last.from_target
        );
        gallery.settle();
        let settled = &gallery.last;
        assert_eq!(gallery.steps_left(), 0);
        assert!(
            settled.step >= TRAINING_STEPS_AFTER_CHANGE && settled.step < MAXIMUM_TRAINING_STEPS_AFTER_CHANGE,
            "{name} trained {} steps",
            settled.step
        );
        assert!(settled.loss < 0.000_000_3, "{name} loss {}", settled.loss);
        let error = largest_error(&gallery.machine, &settled.values);
        assert!(error < 0.003, "{name} error {error}");
        assert!(settled.misalignment < 1e-7, "{name} penalty {}", settled.misalignment);
        // Every line of the sketch is a line of photos, within 0.03% of the stage.
        for (edge, slots) in sketch_lines(&gallery.machine.target) {
            let spread = line_spread(&gallery.machine, &settled.values, &slots, edge);
            assert!(spread < 0.0003, "{name}: the {edge:?} edges of rectangles {slots:?} are {spread} apart");
        }
    }
}

#[test]
fn a_nudged_photo_holds_its_row_within_the_slider_distance_and_lets_go_beyond_it() {
    // Contact sheet's second row is slots 4 to 7. Rectangle 5 goes down by a share of the distance at which a
    // photo lets go of its line, PENALTY_STRENGTH reaches, which is the slider's distance: 0.05 of the height.
    let let_go = DEFAULT_HOLD_DISTANCE * 5.0 / 4.0;
    assert_eq!(PENALTY_STRENGTH, 2.0);
    let nudge = |share: f64| {
        let mut gallery = Gallery::preset(Preset::Contact);
        gallery.settle();
        let drawn = gallery.machine.target.rect(5);
        let moved = Rect { top: drawn.top + share * let_go, ..drawn };
        gallery.set_rect(5, moved);
        gallery.settle();
        let row = line_spread(&gallery.machine, &gallery.last.values, &[4, 5, 6, 7], Edge::Top);
        let others = line_spread(&gallery.machine, &gallery.last.values, &[4, 6, 7], Edge::Top);
        let machine = &gallery.machine;
        let target_items = machine.active_target_items();
        let outputs = machine.match_rectangles(&gallery.last.values, &target_items);
        let photo =
            machine.output_rect(&gallery.last.values, outputs[target_items.iter().position(|&s| s == 5).unwrap()]);
        (row, others, moved.top - photo.top)
    };
    // Within the distance, the photo stays in the row: the penalty pulls all four tops equally, so the row meets the
    // photo a quarter of the way, and the photo rests three quarters of the nudge above its rectangle.
    let (row, others, held) = nudge(0.7);
    assert!(row < 0.0001, "the row's tops are {row} apart");
    assert!(others < 0.0001);
    assert!((held - 0.75 * 0.7 * let_go).abs() < 0.001, "photo 5 rests {held} above its rectangle");
    // Beyond it, the photo is on its rectangle and the rest of the row is left in line.
    let (row, others, held) = nudge(1.3);
    assert!((row - 1.3 * let_go).abs() < 0.001, "the row's tops are {row} apart");
    assert!(held.abs() < 0.001, "photo 5 rests {held} from its rectangle");
    assert!(others < 0.0001, "the other tops are {others} apart");
}
