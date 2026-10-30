//! The learned gallery's network, loss and optimizer, after handwired's src/learned-gallery/trainable-geometry.ts,
//! and its three preset sketches and its rule of when training stops, after learned-gallery.ts. The page's pointer
//! editor, DOM and loss chart are not ported: a frame here is a fixed number of training steps.
//!
//! A small network outputs one rectangle per slot. For each of 20 slots, features of the slot's position that are
//! fixed while building, i.e. the position and its sine and cosine at 4 frequencies, go through two ReGLU layers of
//! 6 units to the slot's left, top, width and height. The network's 244 weights are the program's source cells, and
//! training steps them with Adam, using gradients from `Circuit::backpropagate_reference`, until the rectangles
//! match a sketch. A second part of the loss, the misalignment penalty, pulls photo edges of one kind that are
//! nearly on one line onto one line.

// The test includes this file too and uses a different part of it.
#![allow(dead_code)]

use std::f64::consts::PI;

use handwired_rs::{BackpropOptions, Circuit, Lin, TensorProgram, Wire};

/// The network's output slots, and the photos that the page uses of them.
pub const SLOTS: usize = 20;
pub const PHOTOS: usize = 12;
const HIDDEN: usize = 6;
const FREQUENCIES: usize = 4;
const FEATURES: usize = 2 + FREQUENCIES * 2;
/// A rectangle's fields, in the order the network outputs them.
pub const LEFT: usize = 0;
pub const TOP: usize = 1;
pub const WIDTH: usize = 2;
pub const HEIGHT: usize = 3;
const FIELDS: usize = 4;

/// The four kinds of edge of a rectangle: left, right, top and bottom. An edge is the rectangle's position on the
/// kind's axis, plus its size on that axis for a right or a bottom edge.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Edge {
    Left,
    Right,
    Top,
    Bottom,
}

pub const EDGES: [Edge; 4] = [Edge::Left, Edge::Right, Edge::Top, Edge::Bottom];

impl Edge {
    fn fields(self) -> (usize, usize, f64) {
        match self {
            Edge::Left => (LEFT, WIDTH, 0.0),
            Edge::Right => (LEFT, WIDTH, 1.0),
            Edge::Top => (TOP, HEIGHT, 0.0),
            Edge::Bottom => (TOP, HEIGHT, 1.0),
        }
    }

    /// Whether the edge's position is a fraction of the stage's width, and not of its height.
    pub fn across(self) -> bool {
        matches!(self, Edge::Left | Edge::Right)
    }
}

// The loss's first part is the Huber loss of each output edge against its target rectangle's edge: half the squared
// error up to this error, and linear beyond it.
const GEOMETRY_LOSS_THRESHOLD: f64 = 0.08;
// The penalty's pull between two edges of one kind is this times their gap, and at most the reach minus the gap.
const PENALTY_STIFFNESS: f64 = 5.0;
/// The penalty's weight against the distance from the target. An edge joins a line within one reach of it and
/// leaves the line at this many reaches.
pub const PENALTY_STRENGTH: f64 = 2.0;
// After a reset the penalty is off until the distance from the target is below the onset distance, and for the
// onset steps at the most. Then it rises to its strength over the rise steps.
const PENALTY_ONSET_DISTANCE: f64 = 1e-6;
const PENALTY_ONSET_STEPS: u32 = 1000;
const PENALTY_RISE_STEPS: f64 = 200.0;
// Edges of one kind within this many reaches of the next one count as one line.
const LINE_GAP: f64 = 1.0 / 8.0;
// The share of a pair's pull that its multiplier grows by per training step (the method of multipliers).
const MULTIPLIER_GROWTH: f64 = 0.02;

const LEARNING_RATE: f64 = 0.008;
const WEIGHT_DECAY: f64 = 0.000_000_2;
const SEED: u32 = 0x51_7a_9e;
const FIRST_DECAY: f64 = 0.9;
const SECOND_DECAY: f64 = 0.999;

/// The pull between two edges of one kind whose gap is `gap`: positive pulls the first edge down and the second up.
/// `multiplier` is the pair's multiplier, or 0 for the slope of the loss alone.
pub fn pair_pull(gap: f64, multiplier: f64, reach: f64) -> f64 {
    let limit = (reach - gap.abs()).max(0.0);
    (multiplier + PENALTY_STIFFNESS * gap).min(limit).max(-limit)
}

/// `pair_pull` summed from a gap of 0 to `gap`, i.e. the function whose slope `pair_pull` is. With a multiplier of 0
/// it is the charge of the loss.
pub fn pair_cost(gap: f64, multiplier: f64, reach: f64) -> f64 {
    // The mirrored pair has the same sum.
    let end = gap.abs().min(reach);
    let signed_multiplier = if gap < 0.0 { -multiplier } else { multiplier };
    // Up to `rise` the pull is the multiplier plus the stiffness times the gap, and from there on the limit.
    let rise = end.min((reach - signed_multiplier) / (PENALTY_STIFFNESS + 1.0));
    signed_multiplier * rise + 0.5 * PENALTY_STIFFNESS * rise * rise + reach * (end - rise)
        - 0.5 * (end * end - rise * rise)
}

fn robust_loss(error: f64) -> f64 {
    let magnitude = error.abs();
    if magnitude <= GEOMETRY_LOSS_THRESHOLD {
        0.5 * error * error
    } else {
        GEOMETRY_LOSS_THRESHOLD * (magnitude - 0.5 * GEOMETRY_LOSS_THRESHOLD)
    }
}

fn robust_gradient(error: f64) -> f64 {
    error.clamp(-GEOMETRY_LOSS_THRESHOLD, GEOMETRY_LOSS_THRESHOLD)
}

/// Where each group of weights starts in the source table.
#[derive(Copy, Clone)]
struct Layout {
    first_left_weights: usize,
    first_left_biases: usize,
    first_gate_weights: usize,
    first_gate_biases: usize,
    second_left_weights: usize,
    second_left_biases: usize,
    second_gate_weights: usize,
    second_gate_biases: usize,
    output_weights: usize,
    output_biases: usize,
    size: usize,
}

impl Layout {
    fn new() -> Layout {
        let mut offset = 0;
        let mut allocate = |size: usize| {
            offset += size;
            offset - size
        };
        let first_left_weights = allocate(HIDDEN * FEATURES);
        let first_left_biases = allocate(HIDDEN);
        let first_gate_weights = allocate(HIDDEN * FEATURES);
        let first_gate_biases = allocate(HIDDEN);
        let second_left_weights = allocate(HIDDEN * HIDDEN);
        let second_left_biases = allocate(HIDDEN);
        let second_gate_weights = allocate(HIDDEN * HIDDEN);
        let second_gate_biases = allocate(HIDDEN);
        let output_weights = allocate(FIELDS * HIDDEN);
        let output_biases = allocate(FIELDS);
        Layout {
            first_left_weights,
            first_left_biases,
            first_gate_weights,
            first_gate_biases,
            second_left_weights,
            second_left_biases,
            second_gate_weights,
            second_gate_biases,
            output_weights,
            output_biases,
            size: allocate(0),
        }
    }
}

fn weight(start: usize, row: usize, width: usize, column: usize) -> usize {
    start + row * width + column
}

/// `bias + Σ weight × input`, where the weights are source cells and the inputs are computed, so each product is a
/// multiply of two wires.
fn dynamic_projection(circuit: &mut Circuit, inputs: &[Wire], weights: usize, bias: usize, row: usize) -> Wire {
    let mut terms = vec![(Wire::from_index(bias), 1.0)];
    for (column, &input) in inputs.iter().enumerate() {
        let weight = Wire::from_index(weight(weights, row, inputs.len(), column));
        terms.push((circuit.multiply(weight, input), 1.0));
    }
    circuit.affine(Lin::new(terms, 0.0))
}

/// The network as a circuit whose source cells are its weights, and its program with the rectangles as outputs.
pub struct Network {
    pub circuit: Circuit,
    pub program: TensorProgram,
    /// Slot `item`'s field `field` is wire `geometries[item * 4 + field]`.
    pub geometries: Vec<Wire>,
    layout: Layout,
}

impl Network {
    pub fn new() -> Network {
        let layout = Layout::new();
        let mut circuit = Circuit::new(layout.size);
        let mut geometries = Vec::with_capacity(SLOTS * FIELDS);
        for item in 0..SLOTS {
            let position = item as f64 / (SLOTS - 1) as f64 * 2.0 - 1.0;
            let mut features = vec![1.0, position];
            for frequency in 1..=FREQUENCIES {
                let angle = PI * frequency as f64 * position;
                features.push(angle.sin());
                features.push(angle.cos());
            }
            // The features are numbers while building, so each one is a coefficient of a weight.
            let features_of = |weights: usize, bias: usize, neuron: usize| {
                let mut terms: Vec<(Wire, f64)> = features
                    .iter()
                    .enumerate()
                    .map(|(column, &feature)| (Wire::from_index(weight(weights, neuron, FEATURES, column)), feature))
                    .collect();
                terms.push((Wire::from_index(bias + neuron), 1.0));
                Lin::new(terms, 0.0)
            };
            let first: Vec<Wire> = (0..HIDDEN)
                .map(|neuron| {
                    let left = features_of(layout.first_left_weights, layout.first_left_biases, neuron);
                    let gate = features_of(layout.first_gate_weights, layout.first_gate_biases, neuron);
                    circuit.reglu(left, gate)
                })
                .collect();
            let second: Vec<Wire> = (0..HIDDEN)
                .map(|neuron| {
                    let left = dynamic_projection(
                        &mut circuit,
                        &first,
                        layout.second_left_weights,
                        layout.second_left_biases + neuron,
                        neuron,
                    );
                    let gate = dynamic_projection(
                        &mut circuit,
                        &first,
                        layout.second_gate_weights,
                        layout.second_gate_biases + neuron,
                        neuron,
                    );
                    circuit.reglu(left, gate)
                })
                .collect();
            for field in 0..FIELDS {
                let output = dynamic_projection(
                    &mut circuit,
                    &second,
                    layout.output_weights,
                    layout.output_biases + field,
                    field,
                );
                geometries.push(output);
            }
        }
        let program = circuit.compile(Some(&geometries));
        Network { circuit, program, geometries, layout }
    }

    pub fn parameter_count(&self) -> usize {
        self.layout.size
    }

    /// The seeded weights: the same linear congruential generator and scales as handwired's.
    fn initialize(&self, source: &mut [f32]) {
        let layout = &self.layout;
        let mut state = SEED;
        let mut random_signed = |scale: f64| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state as f64 / 4_294_967_296.0) * 2.0 - 1.0) * scale
        };
        let first_scale = (2.0 / (FEATURES + HIDDEN) as f64).sqrt();
        let hidden_scale = (2.0 / (HIDDEN * 2) as f64).sqrt();
        let output_scale = (2.0 / (HIDDEN + FIELDS) as f64).sqrt();
        for neuron in 0..HIDDEN {
            for feature in 0..FEATURES {
                source[weight(layout.first_left_weights, neuron, FEATURES, feature)] =
                    random_signed(first_scale) as f32;
                source[weight(layout.first_gate_weights, neuron, FEATURES, feature)] =
                    random_signed(first_scale) as f32;
            }
            source[layout.first_left_biases + neuron] = random_signed(0.1) as f32;
            source[layout.first_gate_biases + neuron] = (1.0 + random_signed(0.05)) as f32;
            for input in 0..HIDDEN {
                source[weight(layout.second_left_weights, neuron, HIDDEN, input)] = random_signed(hidden_scale) as f32;
                source[weight(layout.second_gate_weights, neuron, HIDDEN, input)] = random_signed(hidden_scale) as f32;
            }
            source[layout.second_left_biases + neuron] = random_signed(0.1) as f32;
            source[layout.second_gate_biases + neuron] = (1.0 + random_signed(0.05)) as f32;
        }
        for field in 0..FIELDS {
            for input in 0..HIDDEN {
                source[weight(layout.output_weights, field, HIDDEN, input)] = random_signed(output_scale * 0.35) as f32;
            }
        }
        source[layout.output_biases + LEFT] = 0.5;
        source[layout.output_biases + TOP] = 0.5;
        source[layout.output_biases + WIDTH] = 0.18;
        source[layout.output_biases + HEIGHT] = 0.18;
    }
}

impl Default for Network {
    fn default() -> Network {
        Network::new()
    }
}

/// A rectangle in fractions of the stage: left and width of its width, top and height of its height.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rect {
    pub left: f64,
    pub top: f64,
    pub width: f64,
    pub height: f64,
}

/// The rectangles that training fits the outputs to, one per active slot, as 32-bit floats like the page's.
#[derive(Clone)]
pub struct Target {
    pub geometries: [f32; SLOTS * FIELDS],
    pub active: [bool; SLOTS],
    /// Which outputs the active targets are fitted with, as many as there are active targets. Without it they are
    /// the first outputs, e.g. outputs 0 to 9 for 10 targets.
    pub outputs: Option<[bool; SLOTS]>,
}

impl Target {
    pub fn new(rects: &[Rect]) -> Target {
        let mut target = Target { geometries: [0.0; SLOTS * FIELDS], active: [false; SLOTS], outputs: None };
        for (item, &rect) in rects.iter().enumerate() {
            target.set(item, rect);
        }
        target
    }

    pub fn set(&mut self, item: usize, rect: Rect) {
        let start = item * FIELDS;
        self.geometries[start + LEFT] = rect.left as f32;
        self.geometries[start + TOP] = rect.top as f32;
        self.geometries[start + WIDTH] = rect.width as f32;
        self.geometries[start + HEIGHT] = rect.height as f32;
        self.active[item] = true;
    }

    pub fn rect(&self, item: usize) -> Rect {
        let start = item * FIELDS;
        Rect {
            left: self.geometries[start + LEFT] as f64,
            top: self.geometries[start + TOP] as f64,
            width: self.geometries[start + WIDTH] as f64,
            height: self.geometries[start + HEIGHT] as f64,
        }
    }

    pub fn active_count(&self) -> usize {
        self.active.iter().filter(|&&active| active).count()
    }

    fn edge(&self, item: usize, edge: Edge) -> f64 {
        let (position, size, far) = edge.fields();
        let start = item * FIELDS;
        self.geometries[start + position] as f64 + far * self.geometries[start + size] as f64
    }
}

/// The minimum-cost assignment of rows to columns (the Hungarian method with potentials), as handwired's: for each
/// row, the column it is assigned.
fn minimum_cost_assignment(costs: &[Vec<f64>]) -> Vec<usize> {
    let rows = costs.len();
    let columns = costs.first().map_or(0, Vec::len);
    assert!(rows <= columns, "rectangle assignment needs enough outputs");
    let mut row_potentials = vec![0.0; rows + 1];
    let mut column_potentials = vec![0.0; columns + 1];
    let mut matched_row_by_column = vec![0usize; columns + 1];
    let mut previous_column = vec![0usize; columns + 1];
    for row in 1..=rows {
        matched_row_by_column[0] = row;
        let mut minimum_costs = vec![f64::INFINITY; columns + 1];
        let mut visited = vec![false; columns + 1];
        let mut column = 0;
        loop {
            visited[column] = true;
            let current_row = matched_row_by_column[column];
            let mut next_column = 0;
            let mut delta = f64::INFINITY;
            for candidate in 1..=columns {
                if visited[candidate] {
                    continue;
                }
                let cost =
                    costs[current_row - 1][candidate - 1] - row_potentials[current_row] - column_potentials[candidate];
                if cost < minimum_costs[candidate] {
                    minimum_costs[candidate] = cost;
                    previous_column[candidate] = column;
                }
                if minimum_costs[candidate] < delta {
                    delta = minimum_costs[candidate];
                    next_column = candidate;
                }
            }
            assert!(delta.is_finite(), "rectangle assignment has no finite solution");
            for candidate in 0..=columns {
                if visited[candidate] {
                    row_potentials[matched_row_by_column[candidate]] += delta;
                    column_potentials[candidate] -= delta;
                } else {
                    minimum_costs[candidate] -= delta;
                }
            }
            column = next_column;
            if matched_row_by_column[column] == 0 {
                break;
            }
        }
        loop {
            let prior = previous_column[column];
            matched_row_by_column[column] = matched_row_by_column[prior];
            column = prior;
            if column == 0 {
                break;
            }
        }
    }
    let mut output_by_target = vec![0; rows];
    for (column, &row) in matched_row_by_column.iter().enumerate().skip(1) {
        if row > 0 {
            output_by_target[row - 1] = column - 1;
        }
    }
    output_by_target
}

/// The outputs and the loss after an evaluation or a frame of training.
#[derive(Clone, Debug)]
pub struct Outcome {
    /// Every wire's value, read with [`Network::geometries`].
    pub values: Vec<f32>,
    pub loss: f64,
    /// The loss's first part, the distance from the target.
    pub from_target: f64,
    /// The loss's second part, the misalignment penalty, times the penalty's weight in this frame.
    pub misalignment: f64,
    /// Every step taken since the last reset, including the steps taken back.
    pub step: u32,
    /// Whether the frame's steps were taken back.
    pub undone: bool,
}

struct Measure {
    from_target: f64,
    misalignment: f64,
    followed: f64,
}

/// The network with its weights, Adam's state and the penalty's state: handwired's trainable geometry machine.
pub struct Machine {
    pub network: Network,
    pub target: Target,
    pub source: Vec<f32>,
    pub initial_source: Vec<f32>,
    first_moment: Vec<f32>,
    second_moment: Vec<f32>,
    training_step: u32,
    // The step count of Adam's bias corrections, which train_or_undo takes back with the moments.
    moment_step: u32,
    // The part of the learning rate that train_or_undo's steps use, from 1/4096 to 1, and how many frames in a row
    // were taken back at 1/64 of the rate or less.
    learning_rate_scale: f64,
    frames_taken_back_at_low_rate: u32,
    // The penalty's reach for each kind of edge, as a fraction of the stage's width for left and right edges and of
    // its height for top and bottom edges. 0 is no penalty.
    reach: [f64; 4],
    // What a frame's steps hold constant, see start_frame: the pairs' multipliers, by kind of edge and by the two
    // target slots, the lower slot first, and for each kind of edge and target slot, the number of edges on the
    // line of the slot's output edge and a number that the edges of one line share.
    multipliers: Vec<f64>,
    line_size: Vec<usize>,
    line_of: Vec<usize>,
    // The penalty's weight in this frame, and the step at which it began to rise after the last reset.
    frame_strength: f64,
    penalty_onset_step: Option<u32>,
}

impl Machine {
    pub fn new(target: Target) -> Machine {
        let network = Network::new();
        let size = network.parameter_count();
        let mut machine = Machine {
            network,
            target,
            source: vec![0.0; size],
            initial_source: vec![0.0; size],
            first_moment: vec![0.0; size],
            second_moment: vec![0.0; size],
            training_step: 0,
            moment_step: 0,
            learning_rate_scale: 1.0,
            frames_taken_back_at_low_rate: 0,
            reach: [0.0; 4],
            multipliers: vec![0.0; EDGES.len() * SLOTS * SLOTS],
            line_size: vec![0; EDGES.len() * SLOTS],
            line_of: vec![0; EDGES.len() * SLOTS],
            frame_strength: 0.0,
            penalty_onset_step: None,
        };
        machine.reset();
        machine
    }

    pub fn training_step(&self) -> u32 {
        self.training_step
    }

    /// The seeded weights, and Adam and the penalty from the start.
    pub fn reset(&mut self) {
        self.network.initialize(&mut self.source);
        self.initial_source.copy_from_slice(&self.source);
        self.first_moment.fill(0.0);
        self.second_moment.fill(0.0);
        // The penalty is off again, and the next frame's start sets every multiplier to 0 with it.
        self.penalty_onset_step = None;
        self.training_step = 0;
        self.moment_step = 0;
        self.learning_rate_scale = 1.0;
        self.frames_taken_back_at_low_rate = 0;
    }

    /// How close two output edges of one kind have to be for the penalty to pull them together: `across` for left
    /// and right edges, as a fraction of the stage's width, and `down` for top and bottom edges, as a fraction of
    /// its height. 0 switches the penalty off.
    pub fn set_penalty_reach(&mut self, across: f64, down: f64) {
        assert!(across >= 0.0 && down >= 0.0, "penalty reach must be nonnegative");
        self.reach = [across, across, down, down];
    }

    pub fn active_target_items(&self) -> Vec<usize> {
        (0..SLOTS).filter(|&item| self.target.active[item]).collect()
    }

    pub fn output_edge(&self, values: &[f32], output_item: usize, edge: Edge) -> f64 {
        let (position, size, far) = edge.fields();
        let start = output_item * FIELDS;
        let geometries = &self.network.geometries;
        values[geometries[start + position].index()] as f64 + far * values[geometries[start + size].index()] as f64
    }

    pub fn output_rect(&self, values: &[f32], output_item: usize) -> Rect {
        let start = output_item * FIELDS;
        let read = |field: usize| values[self.network.geometries[start + field].index()] as f64;
        Rect { left: read(LEFT), top: read(TOP), width: read(WIDTH), height: read(HEIGHT) }
    }

    fn geometry_loss(&self, values: &[f32], target_item: usize, output_item: usize) -> f64 {
        EDGES
            .iter()
            .map(|&edge| robust_loss(self.output_edge(values, output_item, edge) - self.target.edge(target_item, edge)))
            .sum()
    }

    fn active_output_items(&self, count: usize) -> Vec<usize> {
        match &self.target.outputs {
            None => (0..count).collect(),
            Some(outputs) => {
                let items: Vec<usize> = (0..SLOTS).filter(|&item| outputs[item]).collect();
                assert_eq!(items.len(), count, "as many outputs as targets must be active");
                items
            }
        }
    }

    /// For each of `target_items`, the output it is fitted with: the assignment of least distance.
    pub fn match_rectangles(&self, values: &[f32], target_items: &[usize]) -> Vec<usize> {
        let output_items = self.active_output_items(target_items.len());
        let costs: Vec<Vec<f64>> = target_items
            .iter()
            .map(|&target_item| {
                output_items.iter().map(|&output_item| self.geometry_loss(values, target_item, output_item)).collect()
            })
            .collect();
        minimum_cost_assignment(&costs).into_iter().map(|index| output_items[index]).collect()
    }

    /// The output edges of one kind, by the slot of the target that each output is paired with.
    fn read_edges(
        &self,
        values: &[f32],
        target_items: &[usize],
        output_by_target: &[usize],
        edge: Edge,
    ) -> [f64; SLOTS] {
        let mut edges = [0.0; SLOTS];
        for (index, &item) in target_items.iter().enumerate() {
            edges[item] = self.output_edge(values, output_by_target[index], edge);
        }
        edges
    }

    // The number that a pair's charge and pull are divided by: the number of edges on the two edges' lines.
    fn line_count(&self, kind: usize, first: usize, second: usize) -> f64 {
        let first = kind * SLOTS + first;
        let second = kind * SLOTS + second;
        let count = if self.line_of[first] == self.line_of[second] {
            self.line_size[first]
        } else {
            self.line_size[first] + self.line_size[second]
        };
        count as f64
    }

    /// Computes, from the outputs at the start of a frame of `steps` training steps, what the frame's steps hold
    /// constant: the penalty's weight, the multipliers and the line counts.
    fn start_frame(&mut self, values: &[f32], steps: u32) {
        let target_items = self.active_target_items();
        let output_by_target = self.match_rectangles(values, &target_items);
        if self.penalty_onset_step.is_none() {
            let from_target: f64 = target_items
                .iter()
                .zip(&output_by_target)
                .map(|(&target_item, &output_item)| self.geometry_loss(values, target_item, output_item))
                .sum();
            if from_target < PENALTY_ONSET_DISTANCE * (target_items.len() * EDGES.len()) as f64
                || self.training_step >= PENALTY_ONSET_STEPS
            {
                self.penalty_onset_step = Some(self.training_step);
            }
        }
        self.frame_strength = match self.penalty_onset_step {
            None => 0.0,
            Some(onset) => PENALTY_STRENGTH * ((self.training_step - onset) as f64 / PENALTY_RISE_STEPS).min(1.0),
        };
        let mut order = target_items.clone();
        for (kind, &edge) in EDGES.iter().enumerate() {
            let reach = self.reach[kind];
            let edges = self.read_edges(values, &target_items, &output_by_target, edge);
            // Each pair's multiplier grows by a share of the gap's pull, within the pull's limit at this gap, which
            // is 0 while the penalty is off. A pair with a rectangle that is gone has no multiplier.
            for first in 0..SLOTS {
                for second in first + 1..SLOTS {
                    let pair = (kind * SLOTS + first) * SLOTS + second;
                    if !self.target.active[first] || !self.target.active[second] {
                        self.multipliers[pair] = 0.0;
                        continue;
                    }
                    let gap = edges[first] - edges[second];
                    let limit = if self.frame_strength == 0.0 { 0.0 } else { (reach - gap.abs()).max(0.0) };
                    let grown = self.multipliers[pair] + MULTIPLIER_GROWTH * steps as f64 * PENALTY_STIFFNESS * gap;
                    self.multipliers[pair] = grown.min(limit).max(-limit);
                }
            }
            // The lines: the edges by position, where an edge farther than LINE_GAP reaches from the one before it
            // starts a new line.
            order.sort_by(|&a, &b| edges[a].total_cmp(&edges[b]));
            let mut line_start = 0;
            while line_start < order.len() {
                let mut line_end = line_start + 1;
                while line_end < order.len() && edges[order[line_end]] - edges[order[line_end - 1]] <= reach * LINE_GAP
                {
                    line_end += 1;
                }
                for &item in &order[line_start..line_end] {
                    self.line_of[kind * SLOTS + item] = line_start;
                    self.line_size[kind * SLOTS + item] = line_end - line_start;
                }
                line_start = line_end;
            }
        }
    }

    /// The two parts of the loss for these outputs, and `followed`: the function whose slope a frame's steps
    /// follow, which is the loss with each pair's multiplier in its pull, and with a pair beyond the reach counted
    /// at the charge of one reach instead of 0.
    fn measure(&self, values: &[f32]) -> Measure {
        let target_items = self.active_target_items();
        if target_items.is_empty() {
            return Measure { from_target: 0.0, misalignment: 0.0, followed: 0.0 };
        }
        let output_by_target = self.match_rectangles(values, &target_items);
        let from_target: f64 = target_items
            .iter()
            .zip(&output_by_target)
            .map(|(&target_item, &output_item)| self.geometry_loss(values, target_item, output_item))
            .sum();
        let mut misalignment = 0.0;
        let mut followed_penalty = 0.0;
        for (kind, &edge) in EDGES.iter().enumerate() {
            let reach = self.reach[kind];
            let edges = self.read_edges(values, &target_items, &output_by_target, edge);
            for (first_index, &first) in target_items.iter().enumerate() {
                for &second in &target_items[first_index + 1..] {
                    let gap = edges[first] - edges[second];
                    let count = self.line_count(kind, first, second);
                    if gap.abs() < reach {
                        misalignment += pair_cost(gap, 0.0, reach) / count;
                    }
                    let multiplier = self.multipliers[(kind * SLOTS + first) * SLOTS + second];
                    followed_penalty += pair_cost(gap, multiplier, reach) / count;
                }
            }
        }
        let scale = 1.0 / (target_items.len() * EDGES.len()) as f64;
        Measure {
            from_target: from_target * scale,
            misalignment: self.frame_strength * misalignment * scale,
            followed: (from_target + self.frame_strength * followed_penalty) * scale,
        }
    }

    fn evaluate_tensor(&self) -> Vec<f32> {
        let values = self.network.circuit.evaluate_tensor(&self.source, &self.network.program).unwrap();
        assert!(
            self.network.geometries.iter().all(|wire| values[wire.index()].is_finite()),
            "trainable geometry values must be finite"
        );
        values
    }

    fn train_step(&mut self, learning_rate: f64) {
        let values = self.network.circuit.evaluate_reference(&self.source).unwrap();
        let target_items = self.active_target_items();
        if target_items.is_empty() {
            return;
        }
        let output_by_target = self.match_rectangles(&values, &target_items);
        let mut output_gradients = vec![0f32; values.len()];
        let gradient_scale = 1.0 / (target_items.len() * EDGES.len()) as f64;
        let geometries = &self.network.geometries;
        // An edge's slope goes to the output's position, and for a right or a bottom edge to its size too.
        let mut add_edge_gradient = |target_index: usize, edge: Edge, amount: f64| {
            let (position, size, far) = edge.fields();
            let start = output_by_target[target_index] * FIELDS;
            let position_wire = geometries[start + position].index();
            output_gradients[position_wire] = (output_gradients[position_wire] as f64 + amount) as f32;
            if far == 1.0 {
                let size_wire = geometries[start + size].index();
                output_gradients[size_wire] = (output_gradients[size_wire] as f64 + amount) as f32;
            }
        };
        for (kind, &edge) in EDGES.iter().enumerate() {
            for (target_index, &target_item) in target_items.iter().enumerate() {
                let error = self.output_edge(&values, output_by_target[target_index], edge)
                    - self.target.edge(target_item, edge);
                add_edge_gradient(target_index, edge, robust_gradient(error) * gradient_scale);
            }
            let reach = self.reach[kind];
            let edges = self.read_edges(&values, &target_items, &output_by_target, edge);
            for (first_index, &first) in target_items.iter().enumerate() {
                for (second_index, &second) in target_items.iter().enumerate().skip(first_index + 1) {
                    let multiplier = self.multipliers[(kind * SLOTS + first) * SLOTS + second];
                    let pull = pair_pull(edges[first] - edges[second], multiplier, reach);
                    let amount = self.frame_strength * pull / self.line_count(kind, first, second) * gradient_scale;
                    add_edge_gradient(first_index, edge, amount);
                    add_edge_gradient(second_index, edge, -amount);
                }
            }
        }
        let mut gradients = self
            .network
            .circuit
            .backpropagate_reference(&values, &output_gradients, BackpropOptions::default())
            .unwrap();
        self.training_step += 1;
        self.moment_step += 1;
        let first_correction = 1.0 - FIRST_DECAY.powf(self.moment_step as f64);
        let second_correction = 1.0 - SECOND_DECAY.powf(self.moment_step as f64);
        let size = self.source.len();
        let mut magnitude = 0.0;
        for (gradient, &weight) in gradients[..size].iter_mut().zip(&self.source) {
            *gradient = (*gradient as f64 + weight as f64 * WEIGHT_DECAY) as f32;
            magnitude += *gradient as f64 * *gradient as f64;
        }
        let limit = (1.0 / magnitude.sqrt()).min(1.0);
        let moments = self.first_moment.iter_mut().zip(self.second_moment.iter_mut());
        for ((&gradient, weight), (first, second)) in gradients[..size].iter().zip(self.source.iter_mut()).zip(moments)
        {
            let gradient = gradient as f64 * limit;
            *first = (FIRST_DECAY * *first as f64 + (1.0 - FIRST_DECAY) * gradient) as f32;
            *second = (SECOND_DECAY * *second as f64 + (1.0 - SECOND_DECAY) * gradient * gradient) as f32;
            let corrected_first = *first as f64 / first_correction;
            let corrected_second = *second as f64 / second_correction;
            let stepped = *weight as f64 - learning_rate * corrected_first / (corrected_second.sqrt() + 1e-8);
            *weight = (stepped as f32).clamp(-6.0, 6.0);
        }
    }

    /// The outputs with their loss, for a caller that takes no step. It starts a frame of no steps: the penalty
    /// comes on if the outputs are close enough to the target, and the multipliers are brought within their limits
    /// without growing.
    pub fn evaluate(&mut self) -> Outcome {
        let values = self.evaluate_tensor();
        self.start_frame(&values, 0);
        let measure = self.measure(&values);
        Outcome {
            values,
            loss: measure.from_target + measure.misalignment,
            from_target: measure.from_target,
            misalignment: measure.misalignment,
            step: self.training_step,
            undone: false,
        }
    }

    /// Plain Adam steps, each as a frame of its own, and none taken back.
    pub fn train(&mut self, steps: u32) -> Outcome {
        for _ in 0..steps {
            let values = self.evaluate_tensor();
            self.start_frame(&values, 1);
            self.train_step(LEARNING_RATE);
        }
        self.evaluate()
    }

    /// A frame of `steps` steps, taken back when they leave the function they follow higher than it was before
    /// them, so a page never draws a worse fit than the frame before. After a frame that was taken back the steps
    /// use half the learning rate, down to 1/64 of it, and after a kept frame 1.15 times as much, up to the full
    /// rate. When 5 frames in a row at 1/64 were taken back, the rate goes on halving down to 1/4096, and a kept
    /// frame doubles it until it is at 1/64 again.
    pub fn train_or_undo(&mut self, steps: u32) -> Outcome {
        let values_before = self.evaluate_tensor();
        self.start_frame(&values_before, steps);
        let before = self.measure(&values_before);
        let saved_source = self.source.clone();
        let saved_first_moment = self.first_moment.clone();
        let saved_second_moment = self.second_moment.clone();
        let saved_moment_step = self.moment_step;
        for _ in 0..steps {
            self.train_step(LEARNING_RATE * self.learning_rate_scale);
        }
        let mut values = self.evaluate_tensor();
        let mut after = self.measure(&values);
        let undone = after.followed > before.followed;
        if undone {
            self.source = saved_source;
            self.first_moment = saved_first_moment;
            self.second_moment = saved_second_moment;
            self.moment_step = saved_moment_step;
            if self.learning_rate_scale <= 1.0 / 64.0 {
                self.frames_taken_back_at_low_rate += 1;
            }
            let floor = if self.frames_taken_back_at_low_rate >= 5 { 1.0 / 4096.0 } else { 1.0 / 64.0 };
            self.learning_rate_scale = (self.learning_rate_scale * 0.5).max(floor);
            values = self.evaluate_tensor();
            after = before;
        } else {
            let growth = if self.learning_rate_scale < 1.0 / 64.0 { 2.0 } else { 1.15 };
            self.learning_rate_scale = (self.learning_rate_scale * growth).min(1.0);
            if self.learning_rate_scale >= 1.0 / 64.0 {
                self.frames_taken_back_at_low_rate = 0;
            }
        }
        Outcome {
            values,
            loss: after.from_target + after.misalignment,
            from_target: after.from_target,
            misalignment: after.misalignment,
            step: self.training_step,
            undone,
        }
    }
}

// ---- The page's sketches and its rule of when training stops ------------------------------------------------------

/// The page's stage is 5 wide and 4 tall, and a sketch's numbers are fractions of the stage, so a vertical fraction
/// is 5/4 of the horizontal fraction of the same length on screen.
const MARGIN_X: f64 = 0.036;
const GAP_X: f64 = 0.02;
const MARGIN_Y: f64 = MARGIN_X * 5.0 / 4.0;
const GAP_Y: f64 = GAP_X * 5.0 / 4.0;
const LAYOUT_WIDTH: f64 = 1.0 - MARGIN_X * 2.0;
const LAYOUT_HEIGHT: f64 = 1.0 - MARGIN_Y * 2.0;

/// The page's "Misalignment penalty" slider as it loads: how far a rectangle can be dragged from its row or column
/// before its photo lets go of the line, as a fraction of the stage's width. The penalty's reach is half of it.
pub const DEFAULT_HOLD_DISTANCE: f64 = 0.04;
/// After the sketch changes, training runs at least this many steps, then until the photos have come to rest, and
/// at most the maximum.
pub const TRAINING_STEPS_AFTER_CHANGE: u32 = 500;
pub const MAXIMUM_TRAINING_STEPS_AFTER_CHANGE: u32 = 2000;
/// The photos are at rest when no photo edge has moved more than REST_DISTANCE, a fraction of the stage, from where
/// it was REST_STEPS steps or more before.
pub const REST_STEPS: u32 = 100;
pub const REST_DISTANCE: f64 = 0.0001;
/// The page trains 600 steps a second, i.e. 10 steps in a frame of a 60 Hz display. Here every frame is 10 steps.
pub const STEPS_PER_FRAME: u32 = 10;

// The first cells of a grid, row by row.
fn grid_cells(left: f64, top: f64, width: f64, height: f64, columns: usize, rows: usize, cells: usize) -> Vec<Rect> {
    let cell_width = (width - GAP_X * (columns - 1) as f64) / columns as f64;
    let cell_height = (height - GAP_Y * (rows - 1) as f64) / rows as f64;
    (0..cells)
        .map(|cell| Rect {
            left: left + (cell % columns) as f64 * (cell_width + GAP_X),
            top: top + (cell / columns) as f64 * (cell_height + GAP_Y),
            width: cell_width,
            height: cell_height,
        })
        .collect()
}

/// The page's three preset sketches. Each leaves its last cell empty, and edges of one kind are either equal or at
/// least 0.09 of the stage apart.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Preset {
    /// A large rectangle, two beside it, and 2 rows of 4 below.
    Story,
    /// A large rectangle, and a rail of 2 columns and 5 rows beside it.
    Cinema,
    /// 4 columns and 3 rows.
    Contact,
}

impl Preset {
    pub const ALL: [Preset; 3] = [Preset::Story, Preset::Cinema, Preset::Contact];

    pub fn name(self) -> &'static str {
        match self {
            Preset::Story => "Story",
            Preset::Cinema => "Cinema",
            Preset::Contact => "Contact sheet",
        }
    }

    pub fn rects(self) -> Vec<Rect> {
        match self {
            Preset::Story => {
                let (hero_width, hero_height) = (0.6, 0.52);
                let mut rects = vec![Rect { left: MARGIN_X, top: MARGIN_Y, width: hero_width, height: hero_height }];
                rects.extend(grid_cells(
                    MARGIN_X + hero_width + GAP_X,
                    MARGIN_Y,
                    LAYOUT_WIDTH - hero_width - GAP_X,
                    hero_height,
                    1,
                    2,
                    2,
                ));
                rects.extend(grid_cells(
                    MARGIN_X,
                    MARGIN_Y + hero_height + GAP_Y,
                    LAYOUT_WIDTH,
                    LAYOUT_HEIGHT - hero_height - GAP_Y,
                    4,
                    2,
                    7,
                ));
                rects
            }
            Preset::Cinema => {
                let rail_width = 0.13 * 2.0 + GAP_X;
                let hero_width = LAYOUT_WIDTH - rail_width - GAP_X;
                let mut rects = vec![Rect { left: MARGIN_X, top: MARGIN_Y, width: hero_width, height: LAYOUT_HEIGHT }];
                rects.extend(grid_cells(MARGIN_X + hero_width + GAP_X, MARGIN_Y, rail_width, LAYOUT_HEIGHT, 2, 5, 8));
                rects
            }
            Preset::Contact => grid_cells(MARGIN_X, MARGIN_Y, LAYOUT_WIDTH, LAYOUT_HEIGHT, 4, 3, 11),
        }
    }
}

/// The page's training without its pointer editor: a machine whose target is a sketch fitted with the first outputs,
/// the penalty at the slider's default, and frames of STEPS_PER_FRAME steps until the photos rest.
pub struct Gallery {
    pub machine: Machine,
    steps_since_change: u32,
    // The photos' edges as they were after the step rest_step, left, top, right and bottom per output. No edge has
    // moved more than REST_DISTANCE from them since. None after a change, until the next frame.
    rest_edges: [f32; SLOTS * 4],
    rest_step: Option<u32>,
    pub last: Outcome,
}

impl Gallery {
    pub fn new(rects: &[Rect]) -> Gallery {
        let mut target = Target::new(rects);
        let mut outputs = [false; SLOTS];
        outputs[..rects.len()].fill(true);
        target.outputs = Some(outputs);
        let mut machine = Machine::new(target);
        let reach = DEFAULT_HOLD_DISTANCE / PENALTY_STRENGTH;
        machine.set_penalty_reach(reach, reach * 5.0 / 4.0);
        let last = machine.evaluate();
        Gallery { machine, steps_since_change: 0, rest_edges: [0.0; SLOTS * 4], rest_step: None, last }
    }

    pub fn preset(preset: Preset) -> Gallery {
        Gallery::new(&preset.rects())
    }

    fn at_rest(&self) -> bool {
        self.rest_step.is_some_and(|rest_step| self.last.step - rest_step >= REST_STEPS)
    }

    /// How many more steps training may take: none once the photos are at rest and the first steps are over.
    pub fn steps_left(&self) -> u32 {
        if self.at_rest() && self.steps_since_change >= TRAINING_STEPS_AFTER_CHANGE {
            return 0;
        }
        MAXIMUM_TRAINING_STEPS_AFTER_CHANGE.saturating_sub(self.steps_since_change)
    }

    /// A change of the sketch: training goes on from the current weights.
    pub fn set_rect(&mut self, item: usize, rect: Rect) {
        self.machine.target.set(item, rect);
        self.steps_since_change = 0;
        self.rest_step = None;
        self.last = self.machine.evaluate();
    }

    /// One frame of training, or none once training has stopped. Returns whether it trained.
    pub fn frame(&mut self) -> bool {
        let steps = STEPS_PER_FRAME.min(self.steps_left());
        if steps == 0 {
            return false;
        }
        self.last = self.machine.train_or_undo(steps);
        self.steps_since_change += steps;
        // How far the edge that moved most is from where it was at rest_step. A move past REST_DISTANCE starts the
        // count of steps again from where the photos are now.
        let outputs = self.machine.target.outputs.expect("a gallery fits its sketch with chosen outputs");
        let mut moved: f64 = 0.0;
        let mut edges_now = [0f32; SLOTS * 4];
        for item in (0..SLOTS).filter(|&item| outputs[item]) {
            let rect = self.machine.output_rect(&self.last.values, item);
            let edges = [rect.left, rect.top, rect.left + rect.width, rect.top + rect.height];
            for (side, &edge) in edges.iter().enumerate() {
                moved = moved.max((edge - self.rest_edges[item * 4 + side] as f64).abs());
                edges_now[item * 4 + side] = edge as f32;
            }
        }
        if self.rest_step.is_none() || moved > REST_DISTANCE {
            self.rest_edges = edges_now;
            self.rest_step = Some(self.last.step);
        }
        true
    }

    /// Frames until training stops. Returns the frames taken back.
    pub fn settle(&mut self) -> u32 {
        let mut undone = 0;
        while self.frame() {
            undone += u32::from(self.last.undone);
        }
        undone
    }
}

// ---- Measuring a fit ------------------------------------------------------------------------------------------------

/// The lines of a sketch: for each kind of edge, the active slots whose edges of that kind are equal, 2 or more.
pub fn sketch_lines(target: &Target) -> Vec<(Edge, Vec<usize>)> {
    let slots: Vec<usize> = (0..SLOTS).filter(|&slot| target.active[slot]).collect();
    let mut lines = Vec::new();
    for edge in EDGES {
        let mut sorted = slots.clone();
        sorted.sort_by(|&a, &b| target.edge(a, edge).total_cmp(&target.edge(b, edge)));
        let mut line_start = 0;
        while line_start < sorted.len() {
            let mut line_end = line_start + 1;
            // The sketch's numbers are 32-bit floats, so two sums that are equal as drawn can differ in their last
            // digits.
            while line_end < sorted.len()
                && target.edge(sorted[line_end], edge) - target.edge(sorted[line_end - 1], edge) < 1e-6
            {
                line_end += 1;
            }
            if line_end - line_start >= 2 {
                lines.push((edge, sorted[line_start..line_end].to_vec()));
            }
            line_start = line_end;
        }
    }
    lines
}

/// How far apart the photo edges of one kind are for these target slots, as a fraction of the stage.
pub fn line_spread(machine: &Machine, values: &[f32], slots: &[usize], edge: Edge) -> f64 {
    let target_items = machine.active_target_items();
    let output_by_target = machine.match_rectangles(values, &target_items);
    let positions: Vec<f64> = slots
        .iter()
        .map(|slot| {
            let index = target_items.iter().position(|item| item == slot).expect("the slot is in the sketch");
            machine.output_edge(values, output_by_target[index], edge)
        })
        .collect();
    positions.iter().copied().fold(f64::MIN, f64::max) - positions.iter().copied().fold(f64::MAX, f64::min)
}

/// The largest difference between a photo's left, top, width or height and its rectangle's, as a fraction of the
/// stage.
pub fn largest_error(machine: &Machine, values: &[f32]) -> f64 {
    let target_items = machine.active_target_items();
    let output_by_target = machine.match_rectangles(values, &target_items);
    let mut largest: f64 = 0.0;
    for (&target_item, &output_item) in target_items.iter().zip(&output_by_target) {
        let actual = machine.output_rect(values, output_item);
        let expected = machine.target.rect(target_item);
        for (a, e) in [
            (actual.left, expected.left),
            (actual.top, expected.top),
            (actual.width, expected.width),
            (actual.height, expected.height),
        ] {
            largest = largest.max((a - e).abs());
        }
    }
    largest
}
