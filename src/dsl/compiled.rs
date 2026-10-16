//! Parameters, return values, and the compiled program a `#[handwired]`
//! function produces.

use std::cell::Cell;
use std::fmt::Debug;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Once;

use super::{CONTEXT, Kind, Mux, Scalar, Val, is_wide, with_circuit};
use crate::Error;
use crate::batch::BatchExecutor;
use crate::circuit::{BackpropOptions, Circuit};
use crate::tensor::{TensorExecutor, TensorProgram};
use crate::wire::Wire;

/// What a parameter declares with `#[range(lo, hi)]`, `#[ranges([...])]` and `#[grid(step)]`. A parameter of a
/// compound type, e.g. `[u8; 4]`, applies `range` and `grid` to each scalar; `ranges` gives one range per scalar, in
/// [`HwType::encode`] order.
#[derive(Copy, Clone, Debug, Default)]
pub struct Attrs {
    pub range: Option<(f64, f64)>,
    pub grid: Option<f64>,
    pub ranges: Option<&'static [(f64, f64)]>,
}

impl Attrs {
    pub const NONE: Attrs = Attrs { range: None, grid: None, ranges: None };

    /// The attributes of the `cells` scalars starting at `offset` within this value.
    pub fn part(&self, offset: usize, cells: usize) -> Attrs {
        match self.ranges {
            Some(ranges) => Attrs { ranges: Some(&ranges[offset..offset + cells]), ..*self },
            None => *self,
        }
    }

    /// A struct field's attributes: the parameter's per-scalar `ranges` when given, else the field's own
    /// declarations, else the parameter's `range` and `grid`.
    pub fn field(&self, own: &Attrs, offset: usize, cells: usize) -> Attrs {
        if self.ranges.is_some() {
            self.part(offset, cells)
        } else if own.range.is_some() || own.ranges.is_some() || own.grid.is_some() {
            *own
        } else {
            *self
        }
    }

    /// Check that `ranges`, if given, has one range per scalar of a value of `cells` scalars.
    pub fn expect_cells(&self, cells: usize) {
        if let Some(ranges) = self.ranges {
            assert_eq!(ranges.len(), cells, "#[ranges] gives {} ranges for {cells} scalars", ranges.len());
        }
    }

    fn scalar_range(&self) -> Option<(f64, f64)> {
        match self.ranges {
            Some(ranges) => {
                self.expect_cells(1);
                Some(ranges[0])
            }
            None => self.range,
        }
    }
}

/// Hands out source cells while a function's parameters are declared.
pub struct Inputs {
    next: usize,
}

/// A deterministic random source for sampling inputs.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in 0..1.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A type a `#[handwired]` function can take or return.
pub trait HwType: Sized {
    /// The circuit form of the type, e.g. `Val<u8>` for `u8`.
    type Val: Mux;
    const CELLS: usize;
    fn declare(inputs: &mut Inputs, attrs: &Attrs) -> Self::Val;
    fn encode(&self, out: &mut Vec<f32>);
    fn decode(values: &mut dyn Iterator<Item = f32>) -> Self;
    fn outputs(value: &Self::Val, out: &mut Vec<Wire>);
    fn sample(rng: &mut Rng, attrs: &Attrs) -> Self;
    /// A start value for a binding declared without one.
    fn zero() -> Self::Val;
}

fn scalar_range<T: Scalar>(attrs: &Attrs) -> (f64, f64) {
    match (T::KIND, attrs.scalar_range()) {
        (Kind::Bool, _) => (0.0, 1.0),
        (_, Some((low, high))) => {
            assert!(
                low >= T::MIN && high <= T::MAX && low <= high,
                "#[range({low}, {high})] doesn't fit in {}",
                T::NAME
            );
            if T::KIND == Kind::Float { (f32_at_most(low), f32_at_least(high)) } else { (low, high) }
        }
        (Kind::Int, None) if !is_wide::<T>() => (T::MIN, T::MAX),
        (Kind::Int, None) => panic!("a {} parameter needs #[range(lo, hi)] within ±2^24", T::NAME),
        (Kind::Float, None) => panic!("an f32 parameter needs #[range(lo, hi)]"),
    }
}

/// The largest f32 at or below `value`, so a declared range like 1e-3
/// covers every f32 a caller can pass for it.
fn f32_at_most(value: f64) -> f64 {
    let rounded = value as f32;
    (if rounded as f64 > value { rounded.next_down() } else { rounded }) as f64
}

fn f32_at_least(value: f64) -> f64 {
    let rounded = value as f32;
    (if (rounded as f64) < value { rounded.next_up() } else { rounded }) as f64
}

fn declare_scalar<T: Scalar>(inputs: &mut Inputs, attrs: &Attrs) -> Val<T> {
    let (low, high) = scalar_range::<T>(attrs);
    if T::KIND == Kind::Int {
        assert!(low > -16_777_216.0 && high < 16_777_216.0, "a {} parameter's range must lie within ±2^24", T::NAME);
    }
    let index = inputs.next;
    inputs.next += 1;
    with_circuit(|circuit| {
        let wire = circuit.source(index);
        match T::KIND {
            Kind::Bool => {
                circuit.assume_bit(wire);
                circuit.assume_grid(wire, 1.0);
            }
            Kind::Int => {
                circuit.assume_range(wire, low, high);
                circuit.assume_grid(wire, attrs.grid.unwrap_or(1.0).max(1.0));
            }
            Kind::Float => {
                circuit.assume_range(wire, low, high);
                if let Some(grid) = attrs.grid {
                    circuit.assume_grid(wire, grid);
                }
            }
        }
        Val::from_wire(wire)
    })
}

fn sample_scalar<T: Scalar>(rng: &mut Rng, attrs: &Attrs) -> T {
    let (low, high) = scalar_range::<T>(attrs);
    let pick = rng.next_u64() % 8;
    let value = match pick {
        0 => low,
        1 => high,
        2 if low <= 0.0 && high >= 0.0 => 0.0,
        _ => low + rng.unit() * (high - low),
    };
    let snapped = match (T::KIND, attrs.grid) {
        (Kind::Float, None) => value,
        (Kind::Float, Some(grid)) => ((value / grid).round() * grid).clamp(low, high),
        _ => value.round().clamp(low, high),
    };
    T::from_f64(snapped)
}

macro_rules! scalar_types {
    ($($t:ident),*) => {$(
        impl HwType for $t {
            type Val = Val<$t>;
            const CELLS: usize = 1;
            fn declare(inputs: &mut Inputs, attrs: &Attrs) -> Val<$t> {
                declare_scalar::<$t>(inputs, attrs)
            }
            fn encode(&self, out: &mut Vec<f32>) {
                out.push(self.to_f64() as f32);
            }
            fn decode(values: &mut dyn Iterator<Item = f32>) -> $t {
                <$t as Scalar>::from_f64(values.next().expect("too few output values") as f64)
            }
            fn outputs(value: &Val<$t>, out: &mut Vec<Wire>) {
                out.push(value.wire);
            }
            fn sample(rng: &mut Rng, attrs: &Attrs) -> $t {
                sample_scalar::<$t>(rng, attrs)
            }
            fn zero() -> Val<$t> {
                Val::constant(<$t as Scalar>::from_f64(0.0))
            }
        }
    )*};
}
scalar_types!(bool, u8, u16, u32, u64, usize, i8, i16, i32, i64, f32);

impl<T: HwType, const N: usize> HwType for [T; N] {
    type Val = [T::Val; N];
    const CELLS: usize = N * T::CELLS;
    fn declare(inputs: &mut Inputs, attrs: &Attrs) -> Self::Val {
        attrs.expect_cells(<Self as HwType>::CELLS);
        std::array::from_fn(|item| T::declare(inputs, &attrs.part(item * T::CELLS, T::CELLS)))
    }
    fn encode(&self, out: &mut Vec<f32>) {
        for item in self {
            item.encode(out);
        }
    }
    fn decode(values: &mut dyn Iterator<Item = f32>) -> Self {
        std::array::from_fn(|_| T::decode(values))
    }
    fn outputs(value: &Self::Val, out: &mut Vec<Wire>) {
        for item in value {
            T::outputs(item, out);
        }
    }
    fn sample(rng: &mut Rng, attrs: &Attrs) -> Self {
        attrs.expect_cells(<Self as HwType>::CELLS);
        std::array::from_fn(|item| T::sample(rng, &attrs.part(item * T::CELLS, T::CELLS)))
    }
    fn zero() -> Self::Val {
        std::array::from_fn(|_| T::zero())
    }
}

impl HwType for () {
    type Val = ();
    const CELLS: usize = 0;
    fn declare(_: &mut Inputs, _: &Attrs) {}
    fn encode(&self, _: &mut Vec<f32>) {}
    fn decode(_: &mut dyn Iterator<Item = f32>) {}
    fn outputs(_: &(), _: &mut Vec<Wire>) {}
    fn sample(_: &mut Rng, _: &Attrs) {}
    fn zero() {}
}

/// A function's parameter list, as a tuple, with one `Attrs` per parameter.
pub trait Params: Sized {
    const CELLS: usize;
    fn encode(&self, out: &mut Vec<f32>);
    fn sample(rng: &mut Rng, attrs: &[Attrs]) -> Self;
}

impl Params for () {
    const CELLS: usize = 0;
    fn encode(&self, _: &mut Vec<f32>) {}
    fn sample(_: &mut Rng, _: &[Attrs]) {}
}

macro_rules! tuple_types {
    ($(($($name:ident $index:tt),+)),*) => {$(
        // The last component's offset is never read.
        #[allow(unused_assignments)]
        impl<$($name: HwType),+> HwType for ($($name,)+) {
            type Val = ($($name::Val,)+);
            const CELLS: usize = 0 $(+ $name::CELLS)+;
            fn declare(inputs: &mut Inputs, attrs: &Attrs) -> Self::Val {
                attrs.expect_cells(<Self as HwType>::CELLS);
                let mut offset = 0;
                ($({
                    let part = attrs.part(offset, $name::CELLS);
                    offset += $name::CELLS;
                    $name::declare(inputs, &part)
                },)+)
            }
            fn encode(&self, out: &mut Vec<f32>) {
                $(self.$index.encode(out);)+
            }
            fn decode(values: &mut dyn Iterator<Item = f32>) -> Self {
                ($($name::decode(values),)+)
            }
            fn outputs(value: &Self::Val, out: &mut Vec<Wire>) {
                $($name::outputs(&value.$index, out);)+
            }
            fn sample(rng: &mut Rng, attrs: &Attrs) -> Self {
                attrs.expect_cells(<Self as HwType>::CELLS);
                let mut offset = 0;
                ($({
                    let part = attrs.part(offset, $name::CELLS);
                    offset += $name::CELLS;
                    $name::sample(rng, &part)
                },)+)
            }
            fn zero() -> Self::Val {
                ($($name::zero(),)+)
            }
        }
    )*};
}
tuple_types!(
    (A0 0),
    (A0 0, A1 1),
    (A0 0, A1 1, A2 2),
    (A0 0, A1 1, A2 2, A3 3),
    (A0 0, A1 1, A2 2, A3 3, A4 4),
    (A0 0, A1 1, A2 2, A3 3, A4 4, A5 5),
    (A0 0, A1 1, A2 2, A3 3, A4 4, A5 5, A6 6),
    (A0 0, A1 1, A2 2, A3 3, A4 4, A5 5, A6 6, A7 7),
    (A0 0, A1 1, A2 2, A3 3, A4 4, A5 5, A6 6, A7 7, A8 8),
    (A0 0, A1 1, A2 2, A3 3, A4 4, A5 5, A6 6, A7 7, A8 8, A9 9),
    (A0 0, A1 1, A2 2, A3 3, A4 4, A5 5, A6 6, A7 7, A8 8, A9 9, A10 10),
    (A0 0, A1 1, A2 2, A3 3, A4 4, A5 5, A6 6, A7 7, A8 8, A9 9, A10 10, A11 11)
);

macro_rules! tuple_params {
    ($(($($name:ident $index:tt),+)),*) => {$(
        impl<$($name: HwType),+> Params for ($($name,)+) {
            const CELLS: usize = 0 $(+ $name::CELLS)+;
            fn encode(&self, out: &mut Vec<f32>) {
                $(self.$index.encode(out);)+
            }
            fn sample(rng: &mut Rng, attrs: &[Attrs]) -> Self {
                ($($name::sample(rng, &attrs[$index]),)+)
            }
        }
    )*};
}
tuple_params!(
    (A0 0),
    (A0 0, A1 1),
    (A0 0, A1 1, A2 2),
    (A0 0, A1 1, A2 2, A3 3),
    (A0 0, A1 1, A2 2, A3 3, A4 4),
    (A0 0, A1 1, A2 2, A3 3, A4 4, A5 5),
    (A0 0, A1 1, A2 2, A3 3, A4 4, A5 5, A6 6),
    (A0 0, A1 1, A2 2, A3 3, A4 4, A5 5, A6 6, A7 7),
    (A0 0, A1 1, A2 2, A3 3, A4 4, A5 5, A6 6, A7 7, A8 8),
    (A0 0, A1 1, A2 2, A3 3, A4 4, A5 5, A6 6, A7 7, A8 8, A9 9),
    (A0 0, A1 1, A2 2, A3 3, A4 4, A5 5, A6 6, A7 7, A8 8, A9 9, A10 10),
    (A0 0, A1 1, A2 2, A3 3, A4 4, A5 5, A6 6, A7 7, A8 8, A9 9, A10 10, A11 11),
    (A0 0, A1 1, A2 2, A3 3, A4 4, A5 5, A6 6, A7 7, A8 8, A9 9, A10 10, A11 11, A12 12),
    (A0 0, A1 1, A2 2, A3 3, A4 4, A5 5, A6 6, A7 7, A8 8, A9 9, A10 10, A11 11, A12 12, A13 13),
    (A0 0, A1 1, A2 2, A3 3, A4 4, A5 5, A6 6, A7 7, A8 8, A9 9, A10 10, A11 11, A12 12, A13 13, A14 14),
    (A0 0, A1 1, A2 2, A3 3, A4 4, A5 5, A6 6, A7 7, A8 8, A9 9, A10 10, A11 11, A12 12, A13 13, A14 14, A15 15)
);

/// A `#[handwired]` function's circuit, compiled for its outputs, with the
/// original function kept as the oracle.
pub struct Compiled<Args, R> {
    pub circuit: Circuit,
    pub program: TensorProgram,
    pub outputs: Vec<Wire>,
    oracle: fn(Args) -> R,
    attrs: &'static [Attrs],
}

/// Restores the previous build context, also when a build panics.
struct ContextGuard(Option<Option<Circuit>>);

impl Drop for ContextGuard {
    fn drop(&mut self) {
        if let Some(previous) = self.0.take() {
            CONTEXT.with(|context| *context.borrow_mut() = previous);
        }
    }
}

/// Why [`Compiled::verify`] failed.
#[derive(Clone, Debug, PartialEq)]
pub enum VerifyError {
    /// The circuit and the function disagree on these inputs, each written with `{:?}`.
    Mismatch { inputs: String, function: String, circuit: String },
    /// A sampled input is outside its declared range, e.g. from a custom sampler.
    OutOfRange { inputs: String, error: Error },
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VerifyError::Mismatch { inputs, function, circuit } => {
                write!(formatter, "for inputs {inputs}, the function returns {function} and the circuit {circuit}")
            }
            VerifyError::OutOfRange { inputs, error } => write!(formatter, "inputs {inputs}: {error}"),
        }
    }
}

impl std::error::Error for VerifyError {}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Report {
    /// Inputs where the circuit matched the function.
    pub checked: usize,
    /// Inputs where the function panicked, e.g. on overflow, so any circuit value is allowed.
    pub skipped: usize,
}

thread_local! {
    static QUIET_PANICS: Cell<bool> = const { Cell::new(false) };
}

/// Silence panic messages from oracle calls on this thread only.
fn install_quiet_hook() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let previous = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            if !QUIET_PANICS.with(Cell::get) {
                previous(info);
            }
        }));
    });
}

impl<Args: Params + Clone + Debug, R: HwType + PartialEq + Debug> Compiled<Args, R> {
    /// Build the circuit: `body` declares the parameters and returns the outputs.
    pub fn build(oracle: fn(Args) -> R, attrs: &'static [Attrs], body: impl FnOnce(&mut Inputs) -> R::Val) -> Self {
        let previous = CONTEXT.with(|context| context.replace(Some(Circuit::new(Args::CELLS))));
        let guard = ContextGuard(Some(previous));
        let output = body(&mut Inputs { next: 0 });
        let circuit = CONTEXT.with(|context| context.borrow_mut().take()).expect("the build context disappeared");
        drop(guard);
        let mut outputs = Vec::new();
        R::outputs(&output, &mut outputs);
        let program = circuit.compile(Some(&outputs));
        Compiled { circuit, program, outputs, oracle, attrs }
    }

    fn source(&self, args: &Args) -> Vec<f32> {
        let mut source = Vec::with_capacity(Args::CELLS);
        args.encode(&mut source);
        source
    }

    /// Run the compiled program.
    pub fn eval(&self, args: &Args) -> Result<R, Error> {
        let source = self.source(args);
        self.circuit.validate_source_ranges(&source)?;
        let mut executor = TensorExecutor::new(&self.program);
        Ok(R::decode(&mut executor.run(&source).iter().copied()))
    }

    /// Run the compiled program on many records at once, vectorized across records and split across all cores.
    pub fn run_batch(&self, records: &[Args]) -> Result<Vec<R>, Error> {
        let mut sources = Vec::with_capacity(records.len() * Args::CELLS);
        for record in records {
            let start = sources.len();
            record.encode(&mut sources);
            self.circuit.validate_source_ranges(&sources[start..])?;
        }
        let outputs = BatchExecutor::run_parallel(&self.program, &sources, None);
        Ok(outputs
            .chunks(self.outputs.len().max(1))
            .take(records.len())
            .map(|record| R::decode(&mut record.iter().copied()))
            .collect())
    }

    /// The gradient of output `output` (an index into the flattened outputs, e.g. 0 for a single `f32` result) with
    /// respect to every source cell, summed over `records` with per-record `weights` (all 1 when `None`), e.g.
    /// dLoss/dOutput for each record. Each record runs the exact reference evaluator and its backward pass; records
    /// are split across all cores. Cells are in [`Params::encode`] order, one per scalar.
    pub fn gradients(
        &self,
        records: &[Args],
        output: usize,
        weights: Option<&[f32]>,
        options: BackpropOptions,
    ) -> Result<Vec<f64>, Error>
    where
        Args: Sync,
    {
        assert!(output < self.outputs.len(), "output {output} is out of range: there are {}", self.outputs.len());
        if let Some(weights) = weights {
            assert_eq!(weights.len(), records.len(), "one weight per record");
        }
        let output_wire = self.outputs[output];
        let threads = std::thread::available_parallelism().map_or(1, |count| count.get()).min(records.len().max(1));
        let chunk = records.len().div_ceil(threads).max(1);
        let partials: Vec<Result<Vec<f64>, Error>> = std::thread::scope(|scope| {
            let handles: Vec<_> = records
                .chunks(chunk)
                .enumerate()
                .map(|(chunk_index, chunk_records)| {
                    scope.spawn(move || {
                        let mut sums = vec![0f64; Args::CELLS];
                        let mut seed = vec![0f32; self.circuit.width()];
                        for (offset, record) in chunk_records.iter().enumerate() {
                            let weight = weights.map_or(1.0, |weights| weights[chunk_index * chunk + offset]);
                            if weight == 0.0 {
                                continue;
                            }
                            let values = self.circuit.evaluate_reference(&self.source(record))?;
                            seed[output_wire.index()] = weight;
                            let gradients = self.circuit.backpropagate_reference(&values, &seed, options)?;
                            for (sum, &gradient) in sums.iter_mut().zip(&gradients[..Args::CELLS]) {
                                *sum += gradient as f64;
                            }
                        }
                        Ok(sums)
                    })
                })
                .collect();
            handles.into_iter().map(|handle| handle.join().expect("a gradient thread panicked")).collect()
        });
        let mut total = vec![0f64; Args::CELLS];
        for partial in partials {
            for (sum, value) in total.iter_mut().zip(partial?) {
                *sum += value;
            }
        }
        Ok(total)
    }

    /// Run the reference evaluator, one operation at a time.
    pub fn eval_reference(&self, args: &Args) -> Result<R, Error> {
        let values = self.circuit.evaluate_reference(&self.source(args))?;
        Ok(R::decode(&mut self.outputs.iter().map(|wire| values[wire.index()])))
    }

    /// Run the original function.
    pub fn oracle(&self, args: Args) -> R {
        (self.oracle)(args)
    }

    /// Compare the compiled program with the original function on `cases`
    /// sampled inputs, including each parameter's range ends. Inputs where
    /// the function panics are skipped. Run it in a build with overflow
    /// checks, e.g. a debug build, so that overflow panics as the contract
    /// assumes instead of wrapping.
    pub fn verify(&self, cases: usize) -> Result<Report, VerifyError> {
        self.verify_seeded(cases, 0x5EED)
    }

    pub fn verify_seeded(&self, cases: usize, seed: u64) -> Result<Report, VerifyError> {
        self.verify_with(cases, seed, |rng| Args::sample(rng, self.attrs))
    }

    /// [`verify`](Self::verify) on inputs from `sample`, e.g. states reached by simulating a game, which uniform
    /// sampling of every parameter's range rarely hits.
    pub fn verify_with(
        &self,
        cases: usize,
        seed: u64,
        mut sample: impl FnMut(&mut Rng) -> Args,
    ) -> Result<Report, VerifyError> {
        if !cfg!(debug_assertions) {
            eprintln!(
                "warning: verify in a build without overflow checks: integer overflow wraps in the function instead \
                 of panicking, and the circuit isn't required to match there"
            );
        }
        install_quiet_hook();
        let mut rng = Rng::new(seed);
        let mut executor = TensorExecutor::new(&self.program);
        let mut report = Report { checked: 0, skipped: 0 };
        for _ in 0..cases {
            let args = sample(&mut rng);
            QUIET_PANICS.with(|quiet| quiet.set(true));
            let expected = panic::catch_unwind(AssertUnwindSafe(|| (self.oracle)(args.clone())));
            QUIET_PANICS.with(|quiet| quiet.set(false));
            let Ok(expected) = expected else {
                report.skipped += 1;
                continue;
            };
            let source = self.source(&args);
            self.circuit
                .validate_source_ranges(&source)
                .map_err(|error| VerifyError::OutOfRange { inputs: format!("{args:?}"), error })?;
            let actual = R::decode(&mut executor.run(&source).iter().copied());
            if actual != expected {
                return Err(VerifyError::Mismatch {
                    inputs: format!("{args:?}"),
                    function: format!("{expected:?}"),
                    circuit: format!("{actual:?}"),
                });
            }
            report.checked += 1;
        }
        Ok(report)
    }
}
