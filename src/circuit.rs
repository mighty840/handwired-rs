//! The circuit: source cells plus an ordered list of operations over f32.
//!
//! Ported from `Circuit` in handwired's `src/runtime/numeric-ui-core.ts`. A
//! program has four kinds of operation, each appending one number:
//!
//! ```txt
//! affine(left)      = left
//! reglu(left, gate) = f(left × max(0, gate))
//! boolean(score)    = f(f(f(score) + 8_388_609) - 8_388_609)
//! index(form, table, below, above) = values[table[floor(form)]], with a rule outside the table
//! ```
//!
//! where `left`, `gate` and `form` are affine forms evaluated term by term in
//! stored order, rounding to f32 after every product and sum.

use std::collections::HashMap;

use crate::Error;
use crate::numeric::{
    EXACT_GRID_STEPS, Range, evaluate_boolean, fround, is_power_of_two_grid, largest_grid, lowest_power_of_two, relu32,
    smallest_grid,
};
use crate::wire::{Lin, Term, Wire};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum OpKind {
    Affine,
    Reglu,
    Boolean,
    Index,
}

/// The builder operation a wire came from, so the backward pass can
/// differentiate the whole product or select instead of its ReLUs one by one.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Macro {
    Multiply { left: Wire, right: Wire },
    Select { gate: Wire, off: Wire, on: Wire },
}

/// What an index read returns when floor(index) is outside the table.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum OutOfRange {
    /// The nearest end of the table.
    Clamp,
    /// +0.
    Zero,
}

/// Why an operation's FP32 evaluation, term order or operand wires carry
/// meaning. See handwired's docs/execution-model.md, "Rounding points".
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RoundingPointKind {
    Boolean,
    IntegerBit,
    SignCascade,
    Marked,
}

#[derive(Clone, Debug)]
pub struct IndexRead {
    pub table: usize,
    pub below: OutOfRange,
    pub above: OutOfRange,
    /// The index range the builder accepted.
    pub index_range: Range,
}

/// Exact bits of which exactly one is 1, or, with `allow_all_zero`, at most one. A promise from the caller, like a
/// range, checkable with [`Circuit::validate_one_hot_groups`]. Compilation lowers a sum of products of the group's
/// members with candidate values to a gather: a read of the selected candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct OneHotGroup {
    pub members: Vec<Wire>,
    pub allow_all_zero: bool,
}

#[derive(Clone, Debug)]
pub struct CandidateTable {
    pub candidates: Vec<Wire>,
    pub maximum_depth: u32,
}

#[derive(Clone, Debug)]
pub struct Operation {
    pub kind: OpKind,
    pub left_terms: Vec<Term>,
    pub left_bias: f64,
    /// For an index read, the gate side holds the index form.
    pub gate_terms: Vec<Term>,
    pub gate_bias: f64,
    pub depth: u32,
    pub preserve_rounding: bool,
    pub macro_op: Option<Macro>,
    pub index_read: Option<IndexRead>,
    pub rounding_point: Option<RoundingPointKind>,
    /// Set on the bit that `emit_compare` returns.
    pub comparison_bit: bool,
    /// On that bit: the comparison it decides, for [`ComparisonGradient::Surrogate`].
    pub comparison: Option<Comparison>,
}

/// A comparison bit's test: `d > 0`, `d >= 0` or `d == 0` for the difference `d = bias + Σ coefficient × operand`.
#[derive(Clone, Debug, PartialEq)]
pub struct Comparison {
    pub terms: Vec<Term>,
    pub bias: f64,
    pub equal: bool,
}

/// Gradient rule at a boolean atom, which is a step function.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum BooleanGradient {
    #[default]
    Reject,
    StraightThrough,
}

/// Gradient rule for the index of an index read, which is a step function.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum IndexGradient {
    #[default]
    Reject,
    Zero,
    BackwardDifference,
}

/// Gradient rule for the bit an [`emit_compare`](crate::emit_compare) returns, a step function of its operands.
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub enum ComparisonGradient {
    /// Differentiate the comparison's operations as built: zero almost everywhere, and a boolean atom inside it
    /// follows `boolean_gradient`.
    #[default]
    Exact,
    /// Treat the bit as `sigmoid(d / width)` in the backward pass only, where `d` is the comparison's difference,
    /// e.g. `a - b` for `a > b`; an equality bit as `4 × sigmoid × (1 - sigmoid)`, which peaks at `d = 0`. The
    /// gradient goes straight to the operands; the forward values stay exact. `width` is in the operands' units.
    Surrogate { width: f64 },
}

#[derive(Copy, Clone, Debug, Default)]
pub struct BackpropOptions {
    pub boolean_gradient: BooleanGradient,
    pub index_gradient: IndexGradient,
    pub comparison_gradient: ComparisonGradient,
}

/// Small affine producers with fewer terms than this are copied into the
/// operations that read them; wider sums stay one shared rounding point.
const AFFINE_SHARE_THRESHOLD: usize = 4;

/// Every integer within ±2^24 is an f32 value, so an index range must lie
/// strictly inside it.
const INDEX_MAGNITUDE_LIMIT: f64 = 16_777_216.0;

#[derive(Clone, Debug)]
pub struct Circuit {
    source_size: usize,
    pub(crate) operations: Vec<Operation>,
    depths: Vec<u32>,
    constants: Vec<Option<f64>>,
    pub(crate) expression_wires: HashMap<Vec<u64>, Wire>,
    pub(crate) ranges: Vec<Range>,
    grids: Vec<Option<f64>>,
    exact_bits: Vec<bool>,
    has_readers: Vec<bool>,
    pub(crate) tables: Vec<CandidateTable>,
    table_by_content: HashMap<Vec<Wire>, usize>,
    pub(crate) one_hot_groups: Vec<OneHotGroup>,
}

impl Circuit {
    pub fn new(source_size: usize) -> Circuit {
        Circuit {
            source_size,
            operations: Vec::new(),
            depths: vec![0; source_size],
            constants: vec![None; source_size],
            expression_wires: HashMap::new(),
            ranges: vec![Range::UNKNOWN; source_size],
            grids: vec![None; source_size],
            exact_bits: vec![false; source_size],
            has_readers: vec![false; source_size],
            tables: Vec::new(),
            table_by_content: HashMap::new(),
            one_hot_groups: Vec::new(),
        }
    }

    pub fn source_size(&self) -> usize {
        self.source_size
    }

    /// Source cells plus operations: the length of an evaluation's values.
    pub fn width(&self) -> usize {
        self.source_size + self.operations.len()
    }

    pub fn operations(&self) -> &[Operation] {
        &self.operations
    }

    pub fn operation(&self, wire: Wire) -> Option<&Operation> {
        wire.index().checked_sub(self.source_size).and_then(|index| self.operations.get(index))
    }

    pub fn range(&self, wire: Wire) -> Range {
        self.ranges[wire.index()]
    }

    pub fn grid(&self, wire: Wire) -> Option<f64> {
        self.grids[wire.index()]
    }

    pub fn is_exact_bit(&self, wire: Wire) -> bool {
        self.exact_bits[wire.index()]
    }

    /// The value a constant wire holds before rounding, e.g. `Some(3.0)` for `constant(3.0)`.
    pub fn constant_value(&self, wire: Wire) -> Option<f64> {
        self.constants[wire.index()]
    }

    /// The build-time layer of a wire: 0 for a source cell.
    pub fn depth(&self, wire: Wire) -> u32 {
        self.depths[wire.index()]
    }

    pub fn is_non_negative(&self, wire: Wire) -> bool {
        self.ranges[wire.index()].min >= 0.0
    }

    pub(crate) fn assert_wire(&self, wire: Wire) {
        assert!(wire.index() < self.width(), "wire {} is not in the circuit", wire.index());
    }

    // ---- Declarations ---------------------------------------------------------------------------------------------

    /// Declare the range of a wire's values, a promise from the caller.
    /// Panics when it narrows the range of a wire that operations already
    /// read, since they were built with the old range.
    pub fn assume_range(&mut self, wire: Wire, min: f64, max: f64) {
        self.declare_range(wire, min, max, false);
    }

    /// `assume_range` for a wire that operations already read. Those
    /// operations keep the range they were built with.
    pub fn assume_range_allowing_readers(&mut self, wire: Wire, min: f64, max: f64) {
        self.declare_range(wire, min, max, true);
    }

    fn declare_range(&mut self, wire: Wire, min: f64, max: f64, allow_existing_readers: bool) {
        self.assert_wire(wire);
        assert!(min <= max, "numeric range is inverted");
        let current = self.ranges[wire.index()];
        let narrowed = Range::new(current.min.max(min), current.max.min(max));
        assert!(narrowed.min <= narrowed.max, "numeric range contradicts known bounds");
        let narrows = narrowed.min > current.min || narrowed.max < current.max;
        if narrows && self.has_readers[wire.index()] && !allow_existing_readers {
            panic!(
                "operations that already read wire {} were built with its range {}..{}, and declaring {}..{} now \
                 doesn't change them. Declare the range before the first operation that reads the wire",
                wire.index(),
                current.min,
                current.max,
                min,
                max
            );
        }
        self.ranges[wire.index()] = narrowed;
    }

    /// Declare that every value of the wire is a multiple of `step`, a power of two.
    pub fn assume_grid(&mut self, wire: Wire, step: f64) {
        self.declare_grid(wire, step, false);
    }

    pub fn assume_grid_allowing_readers(&mut self, wire: Wire, step: f64) {
        self.declare_grid(wire, step, true);
    }

    fn declare_grid(&mut self, wire: Wire, step: f64, allow_existing_readers: bool) {
        self.assert_wire(wire);
        assert!(is_power_of_two_grid(step), "a grid step must be a power of two from 2^-149 to 2^127, got {step}");
        let current = self.grids[wire.index()];
        let grid = current.unwrap_or(step).max(step);
        if Some(grid) != current && self.has_readers[wire.index()] && !allow_existing_readers {
            panic!(
                "operations that already read wire {} were built with grid {:?}, and declaring the grid {} now \
                 doesn't change them. Declare the grid before the first operation that reads the wire",
                wire.index(),
                current,
                step
            );
        }
        self.grids[wire.index()] = Some(grid);
    }

    /// Declare that the wire holds exactly 0 or 1. Narrows its range to 0..1.
    pub fn assume_bit(&mut self, wire: Wire) {
        self.assume_range(wire, 0.0, 1.0);
        self.exact_bits[wire.index()] = true;
    }

    /// Declare that `members` are exact bits of which exactly one is 1, or, with `allow_all_zero`, at most one.
    /// Narrows every member's range to 0..1, panicking like [`assume_range`](Self::assume_range) when that changes
    /// a wire that operations already read; declare a group before the code that reads its members. Compilation
    /// reads every group, including one declared late.
    pub fn assume_one_hot(&mut self, members: &[Wire], allow_all_zero: bool) {
        self.declare_one_hot(members, allow_all_zero, false);
    }

    /// `assume_one_hot` for members that operations already read.
    pub fn assume_one_hot_allowing_readers(&mut self, members: &[Wire], allow_all_zero: bool) {
        self.declare_one_hot(members, allow_all_zero, true);
    }

    fn declare_one_hot(&mut self, members: &[Wire], allow_all_zero: bool, allow_existing_readers: bool) {
        assert!(!members.is_empty(), "a one-hot group needs at least one member");
        for (index, &member) in members.iter().enumerate() {
            self.assert_wire(member);
            assert!(!members[..index].contains(&member), "one-hot group members must be distinct wires");
            self.declare_range(member, 0.0, 1.0, allow_existing_readers);
        }
        self.one_hot_groups.push(OneHotGroup { members: members.to_vec(), allow_all_zero });
    }

    pub fn one_hot_groups(&self) -> &[OneHotGroup] {
        &self.one_hot_groups
    }

    /// Check evaluated values against every declared one-hot group: each member exactly 0 or 1, and exactly one of
    /// them 1 (or none, for a group that allows all zeros).
    pub fn validate_one_hot_groups(&self, values: &[f32]) -> Result<(), String> {
        for (group_index, group) in self.one_hot_groups.iter().enumerate() {
            let mut ones = 0;
            for &member in &group.members {
                match values[member.index()] {
                    0.0 => {}
                    1.0 => ones += 1,
                    value => {
                        return Err(format!("one-hot group {group_index}: member {} holds {value}", member.index()));
                    }
                }
            }
            if ones > 1 || (ones == 0 && !group.allow_all_zero) {
                return Err(format!("one-hot group {group_index} has {ones} members set"));
            }
        }
        Ok(())
    }

    /// Mark a wire as an exact bit without declaring its range again, for inlining.
    pub(crate) fn mark_exact_bit(&mut self, wire: Wire) {
        self.exact_bits[wire.index()] = true;
    }

    /// Record an operation as a rounding point, e.g. each step of a round chain.
    pub fn mark_rounding_point(&mut self, wire: Wire) {
        self.record_rounding_point(wire, RoundingPointKind::Marked);
    }

    /// Earlier records win, e.g. a boolean atom that a comparison returns stays `Boolean`.
    pub(crate) fn record_rounding_point(&mut self, wire: Wire, kind: RoundingPointKind) {
        self.assert_wire(wire);
        let index = wire.index().checked_sub(self.source_size).expect("a rounding point must be an operation");
        self.operations[index].rounding_point.get_or_insert(kind);
    }

    /// Every rounding point in wire order.
    pub fn rounding_points(&self) -> Vec<(Wire, RoundingPointKind)> {
        self.operations
            .iter()
            .enumerate()
            .filter_map(|(index, operation)| {
                operation.rounding_point.map(|kind| (Wire::from_index(self.source_size + index), kind))
            })
            .collect()
    }

    // ---- Builder --------------------------------------------------------------------------------------------------

    pub fn source(&self, index: usize) -> Wire {
        assert!(index < self.source_size, "source cell {index} is outside the source table");
        Wire::from_index(index)
    }

    /// A wire that always holds f(value). Ordinary affine readers fold it into their bias.
    pub fn constant(&mut self, value: f64) -> Wire {
        self.affine(value)
    }

    /// A weighted sum, e.g. `circuit.affine(width * 0.5 - 26.0)`. Small sums
    /// are copied into the operations that read them.
    pub fn affine(&mut self, form: impl Into<Lin>) -> Wire {
        self.append(OpKind::Affine, form.into(), Lin::default(), false, None)
    }

    /// A weighted sum that keeps its evaluation, term order and operand wires,
    /// including constants, so readers never copy its terms.
    pub fn round_affine(&mut self, form: impl Into<Lin>) -> Wire {
        self.append(OpKind::Affine, form.into(), Lin::default(), true, None)
    }

    /// `left × max(0, gate)`.
    pub fn reglu(&mut self, left: impl Into<Lin>, gate: impl Into<Lin>) -> Wire {
        self.append(OpKind::Reglu, left.into(), gate.into(), false, None)
    }

    /// `max(0, value)`.
    pub fn relu(&mut self, value: impl Into<Lin>) -> Wire {
        self.reglu(1.0, value)
    }

    /// Rounds a score within 0..1 to 0 below 0.5 and to 1 from 0.5 up.
    pub fn boolean(&mut self, score: Wire) -> Wire {
        let range = self.ranges[score.index()];
        assert!(range.min >= 0.0 && range.max <= 1.0, "boolean input must be within zero and one");
        self.append(OpKind::Boolean, Lin::from(score), Lin::default(), false, None)
    }

    /// `left × right`, one ReGLU when either side is declared nonnegative, three otherwise.
    pub fn multiply(&mut self, left: Wire, right: Wire) -> Wire {
        let macro_op = Some(Macro::Multiply { left, right });
        if self.is_non_negative(right) {
            return self.append(OpKind::Reglu, Lin::from(left), Lin::from(right), true, macro_op);
        }
        if self.is_non_negative(left) {
            return self.append(OpKind::Reglu, Lin::from(right), Lin::from(left), true, macro_op);
        }
        let positive = self.append(OpKind::Reglu, Lin::from(left), Lin::from(right), true, None);
        let negative = self.append(OpKind::Reglu, Lin::from(left), -right, true, None);
        self.append(OpKind::Affine, positive - negative, Lin::default(), true, macro_op)
    }

    /// `gate × on + (1 - gate) × off`. Both sides always compute.
    pub fn select(&mut self, gate: Wire, off: Wire, on: Wire) -> Wire {
        let complement = self.round_affine(1.0 - gate);
        let selected_on = self.multiply(gate, on);
        let selected_off = self.multiply(complement, off);
        self.append(
            OpKind::Affine,
            selected_on + selected_off,
            Lin::default(),
            true,
            Some(Macro::Select { gate, off, on }),
        )
    }

    /// `candidates[floor(index)]` as one operation. The index needs a declared
    /// range strictly inside ±2^24 and must hold an integer.
    pub fn index_read(&mut self, index: Wire, candidates: &[Wire], out_of_range: OutOfRange) -> Wire {
        self.assert_wire(index);
        let declared = self.ranges[index.index()];
        self.index_read_affine(Lin::from(index), candidates, out_of_range, out_of_range, Some(declared))
    }

    /// An index read with the index written as an affine form and one rule per side.
    pub fn index_read_affine(
        &mut self,
        index: Lin,
        candidates: &[Wire],
        below: OutOfRange,
        above: OutOfRange,
        declared_index_range: Option<Range>,
    ) -> Wire {
        for &(wire, coefficient) in &index.terms {
            self.assert_wire(wire);
            assert!(coefficient.is_finite(), "index coefficient must be finite");
        }
        assert!(index.bias.is_finite(), "index bias must be finite");
        let table = self.register_candidate_table(candidates);
        let (terms, bias) = self.fold_constant_terms(normalize_terms(&index.terms), index.bias);
        let read_terms = terms.clone();
        let (terms, bias) = self.compose_affine_terms(&terms, bias);
        if terms.is_empty() {
            let constant_index = fround(bias);
            assert!(!constant_index.is_nan(), "a constant index must not be NaN");
            let count = self.tables[table].candidates.len();
            let position = resolve_index_position(constant_index, count, below, above);
            return if position >= 0 { self.tables[table].candidates[position as usize] } else { self.affine(0.0) };
        }
        let mut index_range = evaluate_affine_range(&terms, bias, &self.ranges);
        // A NaN endpoint passes this check and fails the magnitude check below.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if let Some(declared) = declared_index_range {
            index_range = Range::new(index_range.min.max(declared.min), index_range.max.min(declared.max));
            assert!(
                !(index_range.min > index_range.max),
                "the declared index range contradicts the range inferred from the index form"
            );
        }
        assert!(
            index_range.min > -INDEX_MAGNITUDE_LIMIT && index_range.max < INDEX_MAGNITUDE_LIMIT,
            "index read needs an index range declared strictly inside ±2^24, got {}..{}",
            index_range.min,
            index_range.max
        );
        let mut key = expression_key(OpKind::Index, &[], 0.0, &terms, bias);
        key.extend([u64::MAX, below as u64, above as u64, table as u64]);
        if let Some(&existing) = self.expression_wires.get(&key) {
            return existing;
        }
        let mut depth = self.tables[table].maximum_depth;
        for &(dependency, _) in &terms {
            depth = depth.max(self.depths[dependency.index()]);
        }
        let wire = Wire::from_index(self.width());
        let range = index_read_range(&self.tables[table], index_range, below, above, &self.ranges);
        self.operations.push(Operation {
            kind: OpKind::Index,
            left_terms: Vec::new(),
            left_bias: 0.0,
            gate_terms: terms,
            gate_bias: bias,
            depth: depth + 1,
            preserve_rounding: false,
            macro_op: None,
            index_read: Some(IndexRead { table, below, above, index_range }),
            rounding_point: None,
            comparison_bit: false,
            comparison: None,
        });
        self.push_wire_metadata(depth + 1, None, None, range);
        for &(read, _) in &read_terms {
            self.has_readers[read.index()] = true;
        }
        self.expression_wires.insert(key, wire);
        wire
    }

    fn register_candidate_table(&mut self, candidates: &[Wire]) -> usize {
        assert!(!candidates.is_empty(), "an index read needs at least one candidate");
        if let Some(&table) = self.table_by_content.get(candidates) {
            return table;
        }
        let mut maximum_depth = 0;
        for &candidate in candidates {
            self.assert_wire(candidate);
            maximum_depth = maximum_depth.max(self.depths[candidate.index()]);
            self.has_readers[candidate.index()] = true;
        }
        let table = self.tables.len();
        self.tables.push(CandidateTable { candidates: candidates.to_vec(), maximum_depth });
        self.table_by_content.insert(candidates.to_vec(), table);
        table
    }

    /// Append one operation, or return an existing wire that computes the same
    /// expression. Without `preserve_rounding`, terms are sorted and merged,
    /// constants fold into the bias, small affine producers are copied in, and
    /// a ReGLU with a constant gate becomes an affine.
    pub(crate) fn append(
        &mut self,
        mut kind: OpKind,
        left: Lin,
        gate: Lin,
        preserve_rounding: bool,
        macro_op: Option<Macro>,
    ) -> Wire {
        debug_assert!(kind != OpKind::Index);
        for &(wire, _) in left.terms.iter().chain(&gate.terms) {
            self.assert_wire(wire);
        }
        let (mut left_terms, mut left_bias) = (left.terms, left.bias);
        let (mut gate_terms, mut gate_bias) = (gate.terms, gate.bias);
        let read_left_terms;
        let read_gate_terms;
        if !preserve_rounding {
            (left_terms, left_bias) = self.fold_constant_terms(normalize_terms(&left_terms), left_bias);
            (gate_terms, gate_bias) = self.fold_constant_terms(normalize_terms(&gate_terms), gate_bias);
            read_left_terms = left_terms.clone();
            read_gate_terms = gate_terms.clone();
            if kind == OpKind::Affine
                && let Some(alias) = identity_wire(&left_terms, left_bias)
            {
                return alias;
            }
            (left_terms, left_bias) = self.compose_affine_terms(&left_terms, left_bias);
            (gate_terms, gate_bias) = self.compose_affine_terms(&gate_terms, gate_bias);
            if kind == OpKind::Reglu && gate_terms.is_empty() {
                let gate_value = if gate_bias.is_nan() { f64::NAN } else { gate_bias.max(0.0) };
                kind = OpKind::Affine;
                if gate_value == 0.0 {
                    left_terms.clear();
                } else {
                    for term in &mut left_terms {
                        term.1 *= gate_value;
                    }
                }
                left_bias *= gate_value;
                gate_bias = 0.0;
            } else if kind == OpKind::Reglu && left_terms.is_empty() && left_bias == 0.0 {
                kind = OpKind::Affine;
                gate_terms.clear();
                gate_bias = 0.0;
            }
            if kind == OpKind::Affine
                && let Some(alias) = identity_wire(&left_terms, left_bias)
            {
                return alias;
            }
        } else {
            read_left_terms = left_terms.clone();
            read_gate_terms = gate_terms.clone();
        }
        let mut key = expression_key(kind, &left_terms, left_bias, &gate_terms, gate_bias);
        if preserve_rounding {
            key.push(u64::MAX - 1);
        }
        match macro_op {
            Some(Macro::Multiply { left, right }) => key.extend([u64::MAX - 2, left.0 as u64, right.0 as u64]),
            Some(Macro::Select { gate, off, on }) => {
                key.extend([u64::MAX - 3, gate.0 as u64, off.0 as u64, on.0 as u64])
            }
            None => {}
        }
        if let Some(&existing) = self.expression_wires.get(&key) {
            return existing;
        }
        let wire = Wire::from_index(self.width());
        let gate_dependencies: &[Term] = if kind == OpKind::Reglu { &gate_terms } else { &[] };
        let depth = left_terms
            .iter()
            .chain(gate_dependencies)
            .map(|&(dependency, _)| self.depths[dependency.index()])
            .max()
            .unwrap_or(0);
        let constant =
            if !preserve_rounding && kind == OpKind::Affine && left_terms.is_empty() { Some(left_bias) } else { None };
        let grid =
            infer_operation_grid(kind, &left_terms, left_bias, &gate_terms, gate_bias, &self.grids, &self.ranges);
        let left_range = evaluate_affine_range(&left_terms, left_bias, &self.ranges);
        let range = match kind {
            OpKind::Affine => left_range,
            OpKind::Reglu => {
                multiply_by_relu_range(left_range, evaluate_affine_range(&gate_terms, gate_bias, &self.ranges))
            }
            _ => Range::new(0.0, 1.0),
        };
        if !left_terms.is_empty() {
            for &(read, _) in &read_left_terms {
                self.has_readers[read.index()] = true;
            }
        }
        if !gate_terms.is_empty() {
            for &(read, _) in &read_gate_terms {
                self.has_readers[read.index()] = true;
            }
        }
        self.operations.push(Operation {
            kind,
            left_terms,
            left_bias,
            gate_terms,
            gate_bias,
            depth: depth + 1,
            preserve_rounding,
            macro_op,
            index_read: None,
            rounding_point: if kind == OpKind::Boolean { Some(RoundingPointKind::Boolean) } else { None },
            comparison_bit: false,
            comparison: None,
        });
        self.push_wire_metadata(depth + 1, constant, grid, range);
        self.expression_wires.insert(key, wire);
        wire
    }

    fn push_wire_metadata(&mut self, depth: u32, constant: Option<f64>, grid: Option<f64>, range: Range) {
        self.depths.push(depth);
        self.constants.push(constant);
        self.grids.push(grid);
        self.exact_bits.push(false);
        self.has_readers.push(false);
        self.ranges.push(range);
    }

    fn fold_constant_terms(&self, terms: Vec<Term>, mut bias: f64) -> (Vec<Term>, f64) {
        let mut dynamic_terms = Vec::with_capacity(terms.len());
        for (wire, coefficient) in terms {
            match self.constants[wire.index()] {
                Some(constant) => bias += coefficient * constant,
                None => dynamic_terms.push((wire, coefficient)),
            }
        }
        (dynamic_terms, bias)
    }

    fn compose_affine_terms(&self, terms: &[Term], mut bias: f64) -> (Vec<Term>, f64) {
        let mut composed = Vec::with_capacity(terms.len());
        for &(wire, coefficient) in terms {
            let producer = self.operation(wire);
            match producer {
                Some(operation)
                    if operation.kind == OpKind::Affine
                        && !operation.preserve_rounding
                        && operation.left_terms.len() < AFFINE_SHARE_THRESHOLD =>
                {
                    bias += coefficient * operation.left_bias;
                    for &(dependency, dependency_coefficient) in &operation.left_terms {
                        composed.push((dependency, coefficient * dependency_coefficient));
                    }
                }
                _ => composed.push((wire, coefficient)),
            }
        }
        (normalize_terms(&composed), bias)
    }

    // ---- Evaluation -----------------------------------------------------------------------------------------------

    /// Check that every source cell is within its declared range.
    pub fn validate_source_ranges(&self, source: &[f32]) -> Result<(), Error> {
        if source.len() != self.source_size {
            return Err(Error::SourceLength { expected: self.source_size, actual: source.len() });
        }
        for (index, &value) in source.iter().enumerate() {
            let range = self.ranges[index];
            let value = value as f64;
            if !(value >= range.min && value <= range.max) {
                return Err(Error::SourceOutOfRange { wire: Wire::from_index(index), value, range });
            }
        }
        Ok(())
    }

    /// What the program computes: every wire's value, one operation at a time
    /// in stored order. A compiled program must reproduce every live value.
    pub fn evaluate_reference(&self, source: &[f32]) -> Result<Vec<f32>, Error> {
        self.validate_source_ranges(source)?;
        Ok(self.evaluate_unchecked(source))
    }

    pub(crate) fn evaluate_unchecked(&self, source: &[f32]) -> Vec<f32> {
        let mut values = Vec::with_capacity(self.width());
        values.extend_from_slice(source);
        for operation in &self.operations {
            let left = evaluate_terms(&values, &operation.left_terms, operation.left_bias);
            let value = match operation.kind {
                OpKind::Affine => left,
                OpKind::Reglu => left * relu32(evaluate_terms(&values, &operation.gate_terms, operation.gate_bias)),
                OpKind::Boolean => evaluate_boolean(left),
                OpKind::Index => {
                    let read = operation.index_read.as_ref().unwrap();
                    let index = evaluate_terms(&values, &operation.gate_terms, operation.gate_bias);
                    read_indexed_value(&values, &self.tables[read.table].candidates, read, index as f64)
                }
            };
            values.push(value);
        }
        values
    }

    /// Reverse-mode gradients of the intended real arithmetic: `values` from
    /// `evaluate_reference`, `output_gradients` one per wire. ReLUs get slope 0
    /// at their corner; products and selects are differentiated as a whole.
    pub fn backpropagate_reference(
        &self,
        values: &[f32],
        output_gradients: &[f32],
        options: BackpropOptions,
    ) -> Result<Vec<f32>, Error> {
        assert_eq!(values.len(), self.width(), "forward values have the wrong length");
        assert_eq!(output_gradients.len(), self.width(), "output gradients have the wrong length");
        let mut gradients = output_gradients.to_vec();
        for (operation_index, operation) in self.operations.iter().enumerate().rev() {
            let output_wire = self.source_size + operation_index;
            let output_gradient = gradients[output_wire];
            if output_gradient == 0.0 {
                continue;
            }
            if let (ComparisonGradient::Surrogate { width }, Some(comparison)) =
                (options.comparison_gradient, &operation.comparison)
            {
                assert!(width > 0.0 && width.is_finite(), "a surrogate width must be positive and finite");
                let difference = comparison.bias
                    + comparison
                        .terms
                        .iter()
                        .map(|&(wire, coefficient)| coefficient * values[wire.index()] as f64)
                        .sum::<f64>();
                let sigmoid = 1.0 / (1.0 + (-difference / width).exp());
                let slope = if comparison.equal {
                    4.0 * sigmoid * (1.0 - sigmoid) * (1.0 - 2.0 * sigmoid) / width
                } else {
                    sigmoid * (1.0 - sigmoid) / width
                };
                let scaled: Vec<Term> =
                    comparison.terms.iter().map(|&(wire, coefficient)| (wire, coefficient * slope)).collect();
                accumulate(&mut gradients, &scaled, output_gradient);
                continue;
            }
            match operation.macro_op {
                Some(Macro::Multiply { left, right }) => {
                    accumulate(&mut gradients, &[(left, values[right.index()] as f64)], output_gradient);
                    accumulate(&mut gradients, &[(right, values[left.index()] as f64)], output_gradient);
                    continue;
                }
                Some(Macro::Select { gate, off, on }) => {
                    let gate_value = values[gate.index()];
                    accumulate(
                        &mut gradients,
                        &[(off, (1.0 - gate_value) as f64), (on, gate_value as f64)],
                        output_gradient,
                    );
                    let swing = values[on.index()] - values[off.index()];
                    accumulate(&mut gradients, &[(gate, swing as f64)], output_gradient);
                    continue;
                }
                None => {}
            }
            match operation.kind {
                OpKind::Index => {
                    let read = operation.index_read.as_ref().unwrap();
                    if options.index_gradient == IndexGradient::Reject {
                        return Err(Error::IndexGradient { wire: Wire::from_index(output_wire) });
                    }
                    let candidates = &self.tables[read.table].candidates;
                    let index_value = evaluate_terms(values, &operation.gate_terms, operation.gate_bias) as f64;
                    let position = resolve_index_position(index_value, candidates.len(), read.below, read.above);
                    if position >= 0 {
                        accumulate(&mut gradients, &[(candidates[position as usize], 1.0)], output_gradient);
                    }
                    if options.index_gradient == IndexGradient::BackwardDifference && index_value.is_finite() {
                        let floor_index = index_value.floor();
                        let difference = read_indexed_value(values, candidates, read, floor_index) as f64
                            - read_indexed_value(values, candidates, read, floor_index - 1.0) as f64;
                        let scaled = (output_gradient as f64 * difference) as f32;
                        accumulate(&mut gradients, &operation.gate_terms, scaled);
                    }
                }
                OpKind::Boolean => {
                    if options.boolean_gradient != BooleanGradient::StraightThrough {
                        return Err(Error::BooleanGradient { wire: Wire::from_index(output_wire) });
                    }
                    accumulate(&mut gradients, &operation.left_terms, output_gradient);
                }
                OpKind::Affine => accumulate(&mut gradients, &operation.left_terms, output_gradient),
                OpKind::Reglu => {
                    let left = evaluate_terms(values, &operation.left_terms, operation.left_bias);
                    let gate = evaluate_terms(values, &operation.gate_terms, operation.gate_bias);
                    accumulate(&mut gradients, &operation.left_terms, output_gradient * relu32(gate));
                    if gate > 0.0 {
                        accumulate(&mut gradients, &operation.gate_terms, output_gradient * left);
                    }
                }
            }
        }
        Ok(gradients)
    }
}

// ---- Free helpers ---------------------------------------------------------------------------------------------------

fn key_number(value: f64) -> u64 {
    // JavaScript's keys print -0 as 0, so both share one key.
    (value + 0.0).to_bits()
}

pub(crate) fn expression_key(
    kind: OpKind,
    left_terms: &[Term],
    left_bias: f64,
    gate_terms: &[Term],
    gate_bias: f64,
) -> Vec<u64> {
    let mut key = Vec::with_capacity(4 + 2 * (left_terms.len() + gate_terms.len()));
    key.push(kind as u64);
    key.push(key_number(left_bias));
    for &(wire, coefficient) in left_terms {
        key.extend([wire.0 as u64, key_number(coefficient)]);
    }
    key.push(u64::MAX - 4);
    key.push(key_number(gate_bias));
    for &(wire, coefficient) in gate_terms {
        key.extend([wire.0 as u64, key_number(coefficient)]);
    }
    key
}

fn identity_wire(terms: &[Term], bias: f64) -> Option<Wire> {
    match terms {
        [(wire, coefficient)] if bias == 0.0 && *coefficient == 1.0 => Some(*wire),
        _ => None,
    }
}

/// Sort terms by wire, merge repeated wires, drop zero coefficients.
pub(crate) fn normalize_terms(terms: &[Term]) -> Vec<Term> {
    let mut sorted = terms.to_vec();
    sorted.sort_by_key(|&(wire, _)| wire);
    let mut normalized: Vec<Term> = Vec::with_capacity(sorted.len());
    for (wire, coefficient) in sorted {
        match normalized.last_mut() {
            Some(previous) if previous.0 == wire => previous.1 += coefficient,
            _ => normalized.push((wire, coefficient)),
        }
    }
    normalized.retain(|&(_, coefficient)| coefficient != 0.0);
    normalized
}

/// `f(bias)`, then `value = f(value + f(f(coefficient) × values[wire]))` per term in stored order.
pub(crate) fn evaluate_terms(values: &[f32], terms: &[Term], bias: f64) -> f32 {
    let mut value = bias as f32;
    for &(wire, coefficient) in terms {
        value += coefficient as f32 * values[wire.index()];
    }
    value
}

fn accumulate(gradients: &mut [f32], terms: &[Term], output_gradient: f32) {
    for &(wire, coefficient) in terms {
        gradients[wire.index()] += output_gradient * coefficient as f32;
    }
}

/// floor(index) as a table position: 0..count-1 inside the table; outside it
/// the nearest end, or -1 where that side reads +0; -2 for a NaN index.
pub(crate) fn resolve_index_position(index: f64, count: usize, below: OutOfRange, above: OutOfRange) -> i64 {
    if index.is_nan() {
        return -2;
    }
    let position = index.floor();
    if position < 0.0 {
        return if below == OutOfRange::Zero { -1 } else { 0 };
    }
    if position > (count - 1) as f64 {
        return if above == OutOfRange::Zero { -1 } else { count as i64 - 1 };
    }
    position as i64
}

fn read_indexed_value(values: &[f32], candidates: &[Wire], read: &IndexRead, index: f64) -> f32 {
    match resolve_index_position(index, candidates.len(), read.below, read.above) {
        -2 => f32::NAN,
        -1 => 0.0,
        position => values[candidates[position as usize].index()],
    }
}

pub(crate) fn evaluate_affine_range(terms: &[Term], bias: f64, ranges: &[Range]) -> Range {
    let mut min = fround(bias);
    let mut max = fround(bias);
    for &(wire, coefficient) in terms {
        let range = ranges[wire.index()];
        let rounded = fround(coefficient);
        if rounded > 0.0 {
            min = fround(min + fround(rounded * range.min));
            max = fround(max + fround(rounded * range.max));
        } else {
            min = fround(min + fround(rounded * range.max));
            max = fround(max + fround(rounded * range.min));
        }
        if min.is_nan() || max.is_nan() {
            return Range::UNKNOWN;
        }
    }
    if min.is_nan() || max.is_nan() {
        return Range::UNKNOWN;
    }
    Range::new(min, max)
}

fn multiply_by_relu_range(left: Range, gate: Range) -> Range {
    let gate_min = gate.min.max(0.0);
    let gate_max = gate.max.max(0.0);
    let products = [
        fround(left.min * gate_min),
        fround(left.min * gate_max),
        fround(left.max * gate_min),
        fround(left.max * gate_max),
    ];
    // Zero times an infinite endpoint is NaN; never turn that into a finite bound.
    if products.iter().any(|product| product.is_nan()) {
        return Range::UNKNOWN;
    }
    Range::new(
        products.iter().copied().fold(f64::INFINITY, f64::min),
        products.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    )
}

fn index_read_range(
    table: &CandidateTable,
    index_range: Range,
    below: OutOfRange,
    above: OutOfRange,
    ranges: &[Range],
) -> Range {
    let count = table.candidates.len() as f64;
    let lowest = index_range.min.floor().min(count).max(-1.0) as i64;
    let highest = index_range.max.floor().min(count).max(-1.0) as i64;
    let count = table.candidates.len() as i64;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut include = |range: Range| {
        min = min.min(range.min);
        max = max.max(range.max);
    };
    for position in lowest.max(0)..=highest.min(count - 1) {
        include(ranges[table.candidates[position as usize].index()]);
    }
    let zero = Range::new(0.0, 0.0);
    if lowest < 0 {
        include(if below == OutOfRange::Zero { zero } else { ranges[table.candidates[0].index()] });
    }
    if highest > count - 1 {
        include(if above == OutOfRange::Zero { zero } else { ranges[table.candidates[(count - 1) as usize].index()] });
    }
    Range::new(min, max)
}

struct GridForm {
    grid: f64,
    magnitude: f64,
}

/// The grid of an affine form evaluated in FP32, when every term reads a wire
/// with a grid and the declared ranges keep every product and partial sum
/// within 2^24 grid steps, so nothing rounds.
fn infer_form_grid(terms: &[Term], bias: f64, grids: &[Option<f64>], ranges: &[Range]) -> Option<GridForm> {
    let rounded_bias = fround(bias);
    let mut grid = if rounded_bias == 0.0 { f64::INFINITY } else { lowest_power_of_two(rounded_bias) };
    let mut magnitude = rounded_bias.abs();
    for &(wire, coefficient) in terms {
        let wire_grid = grids[wire.index()]?;
        let range = ranges[wire.index()];
        let largest_operand = range.min.abs().max(range.max.abs());
        if !largest_operand.is_finite() {
            return None;
        }
        let rounded = fround(coefficient);
        if rounded == 0.0 {
            continue;
        }
        grid = grid.min(wire_grid * lowest_power_of_two(rounded));
        magnitude += rounded.abs() * largest_operand;
    }
    if !(grid.is_finite() && grid >= smallest_grid() && magnitude <= EXACT_GRID_STEPS * grid) {
        return None;
    }
    Some(GridForm { grid: grid.min(largest_grid()), magnitude })
}

fn infer_operation_grid(
    kind: OpKind,
    left_terms: &[Term],
    left_bias: f64,
    gate_terms: &[Term],
    gate_bias: f64,
    grids: &[Option<f64>],
    ranges: &[Range],
) -> Option<f64> {
    if kind == OpKind::Boolean {
        return Some(1.0);
    }
    let left = infer_form_grid(left_terms, left_bias, grids, ranges)?;
    if kind == OpKind::Affine {
        return Some(left.grid);
    }
    let gate = infer_form_grid(gate_terms, gate_bias, grids, ranges)?;
    let grid = left.grid * gate.grid;
    if !(grid >= smallest_grid() && left.magnitude * gate.magnitude <= EXACT_GRID_STEPS * grid) {
        return None;
    }
    Some(grid.min(largest_grid()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_affine_sums_compose_and_constants_fold() {
        let mut circuit = Circuit::new(2);
        let (a, b) = (circuit.source(0), circuit.source(1));
        let three = circuit.constant(3.0);
        let sum = circuit.affine(a + b * 2.0);
        let shifted = circuit.affine(sum - three);
        let operation = circuit.operation(shifted).unwrap();
        assert_eq!(operation.left_terms, vec![(a, 1.0), (b, 2.0)]);
        assert_eq!(operation.left_bias, -3.0);
        // The identical expression returns the same wire.
        assert_eq!(circuit.affine(a + b * 2.0), sum);
        // x × 1 + 0 is x itself.
        assert_eq!(circuit.affine(a), a);
    }

    #[test]
    fn ranges_and_grids_are_inferred() {
        let mut circuit = Circuit::new(1);
        let x = circuit.source(0);
        circuit.assume_range(x, 0.0, 16_384.0);
        circuit.assume_grid(x, 1.0 / 64.0);
        let half = circuit.affine(x * 0.5 - 26.0);
        assert_eq!(circuit.range(half), Range::new(-26.0, 8_166.0));
        assert_eq!(circuit.grid(half), Some(1.0 / 128.0));
    }

    #[test]
    #[should_panic(expected = "already read wire 0")]
    fn late_range_declaration_is_refused() {
        let mut circuit = Circuit::new(1);
        let x = circuit.source(0);
        circuit.relu(x);
        circuit.assume_range(x, 0.0, 1.0);
    }

    #[test]
    fn index_read_floors_and_applies_side_rules() {
        let mut circuit = Circuit::new(1);
        let index = circuit.source(0);
        circuit.assume_range(index, -2.0, 10.0);
        let table: Vec<Wire> = [28.0, 44.0, 28.0, 60.0].iter().map(|&value| circuit.constant(value)).collect();
        let clamped = circuit.index_read(index, &table, OutOfRange::Clamp);
        let zeroed = circuit.index_read(index, &table, OutOfRange::Zero);
        assert_eq!(circuit.range(zeroed), Range::new(0.0, 60.0));
        for (input, expected_clamped, expected_zero) in
            [(2.5, 28.0, 28.0), (1.0, 44.0, 44.0), (-1.0, 28.0, 0.0), (7.0, 60.0, 0.0)]
        {
            let values = circuit.evaluate_reference(&[input]).unwrap();
            assert_eq!(values[clamped.index()], expected_clamped);
            assert_eq!(values[zeroed.index()], expected_zero);
        }
        // A constant index selects while building.
        let constant_index =
            circuit.index_read_affine(Lin::constant(3.0), &table, OutOfRange::Clamp, OutOfRange::Clamp, None);
        assert_eq!(constant_index, table[3]);
    }

    #[test]
    fn source_ranges_are_checked() {
        let mut circuit = Circuit::new(1);
        let x = circuit.source(0);
        circuit.assume_range(x, 0.0, 1.0);
        assert!(matches!(circuit.evaluate_reference(&[2.0]), Err(Error::SourceOutOfRange { .. })));
        assert!(matches!(circuit.evaluate_reference(&[f32::NAN]), Err(Error::SourceOutOfRange { .. })));
    }
}
