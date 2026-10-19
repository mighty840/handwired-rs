//! Exact programs as neural nets: a Rust port of the core of
//! [handwired](https://github.com/chenglou/handwired).
//!
//! A program is a fixed list of f32 operations, mostly weighted sums and
//! products with a ReLU, written with builder code that runs once. Because it
//! uses only these primitives, it compiles into layers of sparse matrix
//! multiplications and stays differentiable, while computing exactly what the
//! equivalent ordinary code computes.
//!
//! ```
//! use handwired_rs::{emit_compare, Circuit, Relation};
//!
//! // const result = (a > 4) ? 10 : 12, for an integer a within 0..100.
//! let mut circuit = Circuit::new(1);
//! let a = circuit.source(0);
//! circuit.assume_range(a, 0.0, 100.0);
//! circuit.assume_grid(a, 1.0);
//! let four = circuit.constant(4.0);
//! let above_four = emit_compare(&mut circuit, a, Relation::Greater, four);
//! let result = circuit.affine(above_four * -2.0 + 12.0);
//!
//! for value in 0..=100 {
//!     let values = circuit.evaluate_reference(&[value as f32]).unwrap();
//!     assert_eq!(values[result.index()], if value > 4 { 10.0 } else { 12.0 });
//! }
//! ```
//!
//! Builder methods panic when a contract is violated, e.g. a comparison on an
//! operand without a finite declared range, since that is a bug in the
//! program being built. Evaluation returns [`Error`] for bad input data.

mod batch;
mod cells;
mod circuit;
pub mod dsl;
mod emit;
pub mod export;
mod gather;
mod inline;
mod numeric;
mod prefix;
mod springs;
mod tensor;
pub mod timing;
mod wire;

pub use batch::{BatchExecutor, LANES};
pub use cells::{CellSpec, Policy, SourceTable, constant, event, input, input_bit, state, state_bit};
pub use circuit::{
    BackpropOptions, BooleanGradient, CandidateTable, Circuit, Comparison, ComparisonGradient, IndexGradient,
    IndexRead, Macro, OneHotGroup, OpKind, Operation, OutOfRange, RoundingPointKind, SourceRule,
};
pub use emit::{
    CompareOptions, Relation, emit_and, emit_argmax, emit_clamp_exact, emit_compare, emit_compare_with,
    emit_first_active, emit_maximum, emit_maximum_exact, emit_minimum, emit_minimum_exact, emit_not, emit_or,
    emit_relu, emit_round, emit_select, is_proven_bit,
};
pub use export::{PROGRAM_FORMAT_VERSION, PROGRAM_MAGIC, program_checksum};
pub use handwired_rs_macros::handwired;
pub use numeric::Range;
pub use prefix::{brent_kung_prefix, sklansky_prefix};
pub use springs::{
    AnimatedObject, AnimatedObjectOptions, AnimatedSpring, SpringMatrixWires, SpringState, emit_animated_object,
    emit_apply_spring_matrix, emit_spring,
};
pub use tensor::{CompileOptions, GatherCompute, GatherRead, IndexReadRow, TensorExecutor, TensorLayer, TensorProgram};
pub use wire::{Lin, Term, Wire};

#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    SourceLength {
        expected: usize,
        actual: usize,
    },
    SourceOutOfRange {
        wire: Wire,
        value: f64,
        range: Range,
    },
    /// A nonzero gradient reached a boolean atom without `BooleanGradient::StraightThrough`.
    BooleanGradient {
        wire: Wire,
    },
    /// A nonzero gradient reached an index read without an `IndexGradient` rule.
    IndexGradient {
        wire: Wire,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::SourceLength { expected, actual } => {
                write!(formatter, "source table has {actual} cells, expected {expected}")
            }
            Error::SourceOutOfRange { wire, value, range } => write!(
                formatter,
                "source wire {} is {value}, outside its declared range {}..{}",
                wire.index(),
                range.min,
                range.max
            ),
            Error::BooleanGradient { wire } => {
                write!(formatter, "cannot backpropagate through the boolean atom at wire {}", wire.index())
            }
            Error::IndexGradient { wire } => {
                write!(formatter, "cannot backpropagate into the index of the index read at wire {}", wire.index())
            }
        }
    }
}

impl std::error::Error for Error {}
