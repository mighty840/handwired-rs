//! A circuit's backward pass as flat arrays, for evaluators that run it elsewhere, e.g. a GPU shader that runs one
//! record per invocation (`handwired-rs-gpu`'s `GpuGradients`).
//!
//! [`Circuit::gradient_tape`] keeps the operations a gradient of one output can reach, and those whose values the
//! backward rules read, in circuit order, and settles the [`BackpropOptions`] once: a rule that rejects is an error
//! when the tape is made, and a relative surrogate's width is worked out from the declared ranges. An evaluator then
//! runs each record forward through the operations marked [`FORWARD`], seeds the output's gradient, and goes back
//! through the operations marked [`BACKWARD`], as [`GradientTape::backpropagate`] does on the CPU.

use crate::circuit::{
    BackpropOptions, BooleanGradient, Circuit, ComparisonGradient, IndexGradient, Macro, OpKind, OutOfRange,
    evaluate_affine_range,
};
use crate::numeric::{evaluate_boolean, relu32};
use crate::{Error, Wire};

/// Words per operation in [`GradientTape::operations`].
pub const OPERATION_WORDS: usize = 16;

/// [`GradientTape::operations`] word 0: how the operation computes its value.
pub const KIND_AFFINE: u32 = 0;
pub const KIND_RELU: u32 = 1;
pub const KIND_BOOLEAN: u32 = 2;
pub const KIND_INDEX: u32 = 3;

/// Word 8: how the operation passes its gradient back.
/// To its left terms (an affine operation, or a boolean atom passed straight through).
pub const RULE_TERMS: u32 = 0;
/// A ReLU product `left × relu(gate)`: to the left terms by `relu(gate)`, and to the gate terms by `left` where the
/// gate is positive.
pub const RULE_RELU: u32 = 1;
/// A product `a × b` (words 9 and 10).
pub const RULE_MULTIPLY: u32 = 2;
/// A select `gate ? on : off` (words 9, 10 and 11: gate, off, on).
pub const RULE_SELECT: u32 = 3;
/// A comparison bit's surrogate slope, to the comparison's terms (words 9 and 10: start and count in
/// [`GradientTape::terms`]; 12: width; 13: bias; flag [`EQUAL`]).
pub const RULE_SURROGATE: u32 = 4;
/// An index read: to the candidate read, and with [`BACKWARD_DIFFERENCE`] to the index terms.
pub const RULE_INDEX: u32 = 5;
/// Nothing, e.g. a relative surrogate whose difference has a single possible value.
pub const RULE_NONE: u32 = 6;

/// Word 14 flags.
pub const FORWARD: u32 = 1;
pub const BACKWARD: u32 = 2;
/// An equality comparison's surrogate.
pub const EQUAL: u32 = 4;
/// An index read past the start of its table reads +0 instead of the first candidate.
pub const BELOW_ZERO: u32 = 8;
/// An index read past the end of its table reads +0 instead of the last candidate.
pub const ABOVE_ZERO: u32 = 16;
/// [`IndexGradient::BackwardDifference`] for this index read.
pub const BACKWARD_DIFFERENCE: u32 = 32;

/// The backward pass of one output, as arrays of 32-bit words. Wires are numbered within the tape: the first
/// `source_size` are the source cells, the rest are the operations' outputs, in circuit order.
///
/// Each operation is [`OPERATION_WORDS`] words:
///
/// | word | meaning |
/// |---|---|
/// | 0 | kind: [`KIND_AFFINE`], [`KIND_RELU`], [`KIND_BOOLEAN`] or [`KIND_INDEX`] |
/// | 1 | output wire |
/// | 2, 3 | left terms: start and count in `terms` |
/// | 4, 5 | gate terms (an index read's index): start and count |
/// | 6, 7 | left and gate biases, f32 bits |
/// | 8 | backward rule, `RULE_*` |
/// | 9, 10, 11 | the rule's wires, or a surrogate's terms (start, count) |
/// | 12, 13 | a surrogate's width and bias (f32 bits), or an index read's candidates (start and count in `candidates`) |
/// | 14 | flags: [`FORWARD`], [`BACKWARD`], [`EQUAL`], [`BELOW_ZERO`], [`ABOVE_ZERO`], [`BACKWARD_DIFFERENCE`] |
/// | 15 | the operation's wire in the circuit |
///
/// A term is two words: a wire and its coefficient as f32 bits. Every computation is the circuit's FP32 one, as in
/// [`Circuit::evaluate_reference`] and [`Circuit::backpropagate_reference`].
#[derive(Clone, Debug)]
pub struct GradientTape {
    pub source_size: usize,
    /// The tape's wire count.
    pub width: usize,
    /// The output whose gradient the tape computes, as a tape wire.
    pub output: u32,
    pub operations: Vec<u32>,
    pub terms: Vec<u32>,
    /// Index reads' candidates, as tape wires.
    pub candidates: Vec<u32>,
}

impl Circuit {
    /// The backward pass of `output` under `options`, as a [`GradientTape`]. An index read or boolean atom that a
    /// gradient of `output` can reach, under a rule that rejects it, is an error here, whatever the records; on the
    /// CPU, [`Circuit::backpropagate_reference`] only refuses one that a nonzero gradient reaches.
    pub fn gradient_tape(&self, output: Wire, options: BackpropOptions) -> Result<GradientTape, Error> {
        assert!(output.index() < self.width(), "wire {} isn't in the circuit", output.index());
        let sources = self.source_size();
        let operation_of = |wire: usize| wire.checked_sub(sources).map(|index| &self.operations[index]);

        // The rule each operation's gradient follows, as in backpropagate_reference.
        let rules: Vec<Rule> = self
            .operations
            .iter()
            .map(|operation| match (options.comparison_gradient, &operation.comparison) {
                (ComparisonGradient::Surrogate { width }, Some(_)) => {
                    assert!(width > 0.0 && width.is_finite(), "a surrogate width must be positive and finite");
                    Rule::Surrogate(width)
                }
                (ComparisonGradient::RelativeSurrogate { fraction }, Some(comparison)) => {
                    assert!(fraction > 0.0 && fraction.is_finite(), "a surrogate fraction must be positive and finite");
                    let range = evaluate_affine_range(&comparison.terms, comparison.bias, &self.ranges);
                    let span = range.max - range.min;
                    if span > 0.0 && span.is_finite() { Rule::Surrogate(fraction * span) } else { Rule::None }
                }
                _ => match (operation.macro_op, operation.kind) {
                    (Some(Macro::Multiply { .. }), _) => Rule::Multiply,
                    (Some(Macro::Select { .. }), _) => Rule::Select,
                    (None, OpKind::Index) => Rule::Index,
                    (None, OpKind::Reglu) => Rule::Relu,
                    (None, OpKind::Affine | OpKind::Boolean) => Rule::Terms,
                },
            })
            .collect();

        // Back from the output: the operations a gradient reaches, and the wires whose values their rules read.
        let mut reached = vec![false; self.width()];
        let mut needed = vec![false; self.width()];
        reached[output.index()] = true;
        for (index, operation) in self.operations.iter().enumerate().rev() {
            let wire = sources + index;
            if !reached[wire] {
                continue;
            }
            let mut reach = |wire: Wire| reached[wire.index()] = true;
            match rules[index] {
                Rule::Surrogate(_) => {
                    for &(operand, _) in &operation.comparison.as_ref().unwrap().terms {
                        reach(operand);
                        needed[operand.index()] = true;
                    }
                }
                Rule::Multiply => {
                    let Some(Macro::Multiply { left, right }) = operation.macro_op else { unreachable!() };
                    reach(left);
                    reach(right);
                    needed[left.index()] = true;
                    needed[right.index()] = true;
                }
                Rule::Select => {
                    let Some(Macro::Select { gate, off, on }) = operation.macro_op else { unreachable!() };
                    for operand in [gate, off, on] {
                        reach(operand);
                        needed[operand.index()] = true;
                    }
                }
                Rule::Index => {
                    if options.index_gradient == IndexGradient::Reject {
                        return Err(Error::IndexGradient { wire: Wire::from_index(wire) });
                    }
                    let read = operation.index_read.as_ref().unwrap();
                    for &candidate in &self.tables[read.table].candidates {
                        reach(candidate);
                        needed[candidate.index()] = true;
                    }
                    for &(operand, _) in &operation.gate_terms {
                        needed[operand.index()] = true;
                        if options.index_gradient == IndexGradient::BackwardDifference {
                            reach(operand);
                        }
                    }
                }
                Rule::Relu => {
                    for &(operand, _) in operation.left_terms.iter().chain(&operation.gate_terms) {
                        reach(operand);
                        needed[operand.index()] = true;
                    }
                }
                Rule::Terms => {
                    if operation.kind == OpKind::Boolean && options.boolean_gradient != BooleanGradient::StraightThrough
                    {
                        return Err(Error::BooleanGradient { wire: Wire::from_index(wire) });
                    }
                    for &(operand, _) in &operation.left_terms {
                        reach(operand);
                    }
                }
                Rule::None => {}
            }
        }
        // Forward: everything a needed value is computed from.
        for index in (0..self.operations.len()).rev() {
            let wire = sources + index;
            if !needed[wire] {
                continue;
            }
            let operation = &self.operations[index];
            for &(operand, _) in operation.left_terms.iter().chain(&operation.gate_terms) {
                needed[operand.index()] = true;
            }
            if let Some(read) = &operation.index_read {
                for &candidate in &self.tables[read.table].candidates {
                    needed[candidate.index()] = true;
                }
            }
        }

        // Number the kept wires: every source cell, then each kept operation.
        let mut tape_wire = vec![u32::MAX; self.width()];
        for (cell, slot) in tape_wire.iter_mut().enumerate().take(sources) {
            *slot = cell as u32;
        }
        let mut width = sources;
        for wire in sources..self.width() {
            if reached[wire] || needed[wire] {
                tape_wire[wire] = width as u32;
                width += 1;
            }
        }
        let mut tape = GradientTape {
            source_size: sources,
            width,
            output: tape_wire[output.index()],
            operations: Vec::new(),
            terms: Vec::new(),
            candidates: Vec::new(),
        };
        let push_terms = |terms: &mut Vec<u32>, list: &[(Wire, f64)]| {
            let start = terms.len() / 2;
            for &(wire, coefficient) in list {
                terms.extend([tape_wire[wire.index()], (coefficient as f32).to_bits()]);
            }
            (start as u32, list.len() as u32)
        };
        for wire in sources..self.width() {
            if tape_wire[wire] == u32::MAX {
                continue;
            }
            let index = wire - sources;
            let operation = operation_of(wire).unwrap();
            let mut words = [0u32; OPERATION_WORDS];
            words[0] = match operation.kind {
                OpKind::Affine => KIND_AFFINE,
                OpKind::Reglu => KIND_RELU,
                OpKind::Boolean => KIND_BOOLEAN,
                OpKind::Index => KIND_INDEX,
            };
            words[1] = tape_wire[wire];
            (words[2], words[3]) = push_terms(&mut tape.terms, &operation.left_terms);
            (words[4], words[5]) = push_terms(&mut tape.terms, &operation.gate_terms);
            words[6] = (operation.left_bias as f32).to_bits();
            words[7] = (operation.gate_bias as f32).to_bits();
            let mut flags = if needed[wire] { FORWARD } else { 0 } | if reached[wire] { BACKWARD } else { 0 };
            words[8] = match rules[index] {
                Rule::Terms => RULE_TERMS,
                Rule::Relu => RULE_RELU,
                Rule::Multiply => {
                    let Some(Macro::Multiply { left, right }) = operation.macro_op else { unreachable!() };
                    (words[9], words[10]) = (tape_wire[left.index()], tape_wire[right.index()]);
                    RULE_MULTIPLY
                }
                Rule::Select => {
                    let Some(Macro::Select { gate, off, on }) = operation.macro_op else { unreachable!() };
                    (words[9], words[10], words[11]) =
                        (tape_wire[gate.index()], tape_wire[off.index()], tape_wire[on.index()]);
                    RULE_SELECT
                }
                Rule::Surrogate(width) => {
                    let comparison = operation.comparison.as_ref().unwrap();
                    (words[9], words[10]) = push_terms(&mut tape.terms, &comparison.terms);
                    words[12] = (width as f32).to_bits();
                    words[13] = (comparison.bias as f32).to_bits();
                    if comparison.equal {
                        flags |= EQUAL;
                    }
                    RULE_SURROGATE
                }
                Rule::Index => RULE_INDEX,
                Rule::None => RULE_NONE,
            };
            if let Some(read) = &operation.index_read {
                let candidates = &self.tables[read.table].candidates;
                words[12] = tape.candidates.len() as u32;
                words[13] = candidates.len() as u32;
                tape.candidates.extend(candidates.iter().map(|candidate| tape_wire[candidate.index()]));
                flags |= if read.below == OutOfRange::Zero { BELOW_ZERO } else { 0 };
                flags |= if read.above == OutOfRange::Zero { ABOVE_ZERO } else { 0 };
                if options.index_gradient == IndexGradient::BackwardDifference {
                    flags |= BACKWARD_DIFFERENCE;
                }
            }
            words[14] = flags;
            words[15] = wire as u32;
            tape.operations.extend(words);
        }
        Ok(tape)
    }
}

#[derive(Copy, Clone)]
enum Rule {
    Terms,
    Relu,
    Multiply,
    Select,
    Surrogate(f64),
    Index,
    None,
}

impl GradientTape {
    pub fn operation_count(&self) -> usize {
        self.operations.len() / OPERATION_WORDS
    }

    /// The gradient of the output, seeded with `seed`, with respect to each source cell of one record: the
    /// reference for other evaluators of the tape. Equal, bit for bit, to [`Circuit::backpropagate_reference`]'s
    /// source gradients, except under a surrogate rule, whose coefficients, width and bias the tape holds as f32.
    pub fn backpropagate(&self, source: &[f32], seed: f32) -> Vec<f32> {
        assert_eq!(source.len(), self.source_size, "one value per source cell");
        let mut values = vec![0f32; self.width];
        values[..self.source_size].copy_from_slice(source);
        let operations: Vec<&[u32]> = self.operations.chunks(OPERATION_WORDS).collect();
        for words in &operations {
            if words[14] & FORWARD != 0 {
                values[words[1] as usize] = self.forward(words, &values);
            }
        }
        let mut gradients = vec![0f32; self.width];
        gradients[self.output as usize] = seed;
        for words in operations.iter().rev() {
            let gradient = gradients[words[1] as usize];
            if words[14] & BACKWARD == 0 || gradient == 0.0 {
                continue;
            }
            self.backward(words, &values, &mut gradients, gradient);
        }
        gradients.truncate(self.source_size);
        gradients
    }

    fn listed(&self, start: u32, count: u32) -> impl Iterator<Item = (usize, f32)> + '_ {
        self.terms[2 * start as usize..2 * (start + count) as usize]
            .chunks(2)
            .map(|term| (term[0] as usize, f32::from_bits(term[1])))
    }

    fn sum(&self, values: &[f32], start: u32, count: u32, bias: u32) -> f32 {
        let mut value = f32::from_bits(bias);
        for (wire, coefficient) in self.listed(start, count) {
            value += coefficient * values[wire];
        }
        value
    }

    /// An index read's candidate position for `index`: 0..count, or -1 where it reads +0, or -2 for NaN.
    fn position(words: &[u32], index: f32) -> i64 {
        if index.is_nan() {
            return -2;
        }
        let (position, count) = (index.floor() as f64, words[13] as f64);
        if position < 0.0 {
            if words[14] & BELOW_ZERO != 0 { -1 } else { 0 }
        } else if position > count - 1.0 {
            if words[14] & ABOVE_ZERO != 0 { -1 } else { count as i64 - 1 }
        } else {
            position as i64
        }
    }

    fn read(&self, words: &[u32], values: &[f32], index: f32) -> f32 {
        match GradientTape::position(words, index) {
            -2 => f32::NAN,
            -1 => 0.0,
            position => values[self.candidates[(words[12] as i64 + position) as usize] as usize],
        }
    }

    fn forward(&self, words: &[u32], values: &[f32]) -> f32 {
        let left = || self.sum(values, words[2], words[3], words[6]);
        let gate = || self.sum(values, words[4], words[5], words[7]);
        match words[0] {
            KIND_AFFINE => left(),
            KIND_RELU => left() * relu32(gate()),
            KIND_BOOLEAN => evaluate_boolean(left()),
            _ => self.read(words, values, gate()),
        }
    }

    fn backward(&self, words: &[u32], values: &[f32], gradients: &mut [f32], gradient: f32) {
        let mut add = |wire: u32, coefficient: f32| gradients[wire as usize] += gradient * coefficient;
        match words[8] {
            RULE_TERMS => self.listed(words[2], words[3]).for_each(|(wire, c)| add(wire as u32, c)),
            RULE_RELU => {
                let left = self.sum(values, words[2], words[3], words[6]);
                let gate = self.sum(values, words[4], words[5], words[7]);
                let slope = relu32(gate);
                for (wire, coefficient) in self.listed(words[2], words[3]) {
                    gradients[wire] += gradient * slope * coefficient;
                }
                if gate > 0.0 {
                    for (wire, coefficient) in self.listed(words[4], words[5]) {
                        gradients[wire] += gradient * left * coefficient;
                    }
                }
            }
            RULE_MULTIPLY => {
                add(words[9], values[words[10] as usize]);
                add(words[10], values[words[9] as usize]);
            }
            RULE_SELECT => {
                let gate = values[words[9] as usize];
                add(words[10], 1.0 - gate);
                add(words[11], gate);
                add(words[9], values[words[11] as usize] - values[words[10] as usize]);
            }
            RULE_SURROGATE => {
                let width = f32::from_bits(words[12]) as f64;
                let difference = f32::from_bits(words[13]) as f64
                    + self.listed(words[9], words[10]).map(|(wire, c)| c as f64 * values[wire] as f64).sum::<f64>();
                let sigmoid = 1.0 / (1.0 + (-difference / width).exp());
                let slope = if words[14] & EQUAL != 0 {
                    4.0 * sigmoid * (1.0 - sigmoid) * (1.0 - 2.0 * sigmoid) / width
                } else {
                    sigmoid * (1.0 - sigmoid) / width
                };
                for (wire, coefficient) in self.listed(words[9], words[10]) {
                    gradients[wire] += gradient * (coefficient as f64 * slope) as f32;
                }
            }
            RULE_INDEX => {
                let index = self.sum(values, words[4], words[5], words[7]);
                let position = GradientTape::position(words, index);
                if position >= 0 {
                    add(self.candidates[(words[12] as i64 + position) as usize], 1.0);
                }
                if words[14] & BACKWARD_DIFFERENCE != 0 && index.is_finite() {
                    let floor = index.floor();
                    let difference =
                        self.read(words, values, floor) as f64 - self.read(words, values, floor - 1.0) as f64;
                    let scaled = (gradient as f64 * difference) as f32;
                    for (wire, coefficient) in self.listed(words[4], words[5]) {
                        gradients[wire] += scaled * coefficient;
                    }
                }
            }
            _ => {}
        }
    }
}
