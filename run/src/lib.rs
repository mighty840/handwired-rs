//! Run circuits exported by [handwired-rs](https://crates.io/crates/handwired-rs) (`Compiled::export` or
//! `TensorProgram::to_bytes`) without it: `no_std`, no allocation, no dependencies, no `unsafe`. A program is its
//! bytes, e.g. from `include_bytes!` or flash; [`Program::parse`] checks them once, and [`Program::run`] evaluates
//! them into scratch memory the caller provides. Results are bit-exact with handwired-rs's CPU executor wherever f32
//! arithmetic is IEEE 754, including software floating point on chips without an FPU.
//!
//! Every run does the same operations in the same order whatever the inputs, so its time is bounded by the program:
//! it doesn't vary at all where each f32 operation takes fixed time, as on hardware FPUs, and only by how long each
//! operation takes where that depends on the values, as in software floating point. Inputs outside the ranges the
//! function declared, or between the steps of their grids (e.g. 0.5 for an integer), are refused, since the circuit
//! is only exact on them.
//!
//! ```
//! # fn demo(bytes: &[u8]) -> Result<(), handwired_rs_run::Error> {
//! use handwired_rs_run::Program;
//!
//! let program = Program::parse(bytes)?;
//! let mut scratch = [0.0f32; 4096];
//! let mut outputs = [0.0f32; 8];
//! program.run(&[3.0, 1.0], &mut scratch[..program.scratch_len()], &mut outputs[..program.output_count()])?;
//! # Ok(())
//! # }
//! ```

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

use core::fmt;

/// The first 8 bytes of every program.
pub const MAGIC: [u8; 8] = *b"HWCIRC\0\0";
/// The format version this crate reads, and the newest: version 1, which carries no grids, is read too.
pub const FORMAT_VERSION: u32 = 2;

/// Why a program can't be read or run.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// The bytes end before the program does.
    Truncated,
    /// The bytes don't start like a handwired program.
    NotAProgram,
    /// A format version this crate doesn't read.
    UnsupportedVersion(u32),
    /// The checksum doesn't match: the bytes changed after the program was exported.
    Checksum,
    /// The program's parts don't fit together, e.g. an operation reads a value computed after it.
    Malformed(&'static str),
    /// The number of input cells isn't the program's.
    InputLength {
        /// The program's input cells.
        expected: usize,
        /// The cells given.
        actual: usize,
    },
    /// An input cell is outside the range the function declared for it, where the circuit isn't exact.
    InputOutOfRange {
        /// Which cell.
        index: usize,
        /// Its value.
        value: f32,
    },
    /// An input cell isn't a whole multiple of its grid's step, e.g. 0.5 for an integer, where the circuit isn't
    /// exact.
    InputOffGrid {
        /// Which cell.
        index: usize,
        /// Its value.
        value: f32,
        /// The step its values are multiples of.
        step: f64,
    },
    /// The scratch memory is smaller than [`Program::scratch_len`].
    ScratchTooSmall {
        /// The cells needed.
        needed: usize,
        /// The cells given.
        actual: usize,
    },
    /// The number of output cells isn't the program's.
    OutputLength {
        /// The program's output cells.
        expected: usize,
        /// The cells given.
        actual: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Truncated => write!(f, "the program's bytes end early"),
            Error::NotAProgram => write!(f, "not a handwired program"),
            Error::UnsupportedVersion(version) => {
                write!(f, "program format version {version}; this runner reads versions 1 to {FORMAT_VERSION}")
            }
            Error::Checksum => write!(f, "the program's checksum doesn't match its bytes"),
            Error::Malformed(what) => write!(f, "malformed program: {what}"),
            Error::InputLength { expected, actual } => write!(f, "{actual} input cells for a program of {expected}"),
            Error::InputOutOfRange { index, value } => {
                write!(f, "input cell {index} is {value}, outside the range its function declared")
            }
            Error::InputOffGrid { index, value, step } => {
                write!(f, "input cell {index} is {value}, not a multiple of {step}, the step its function declared")
            }
            Error::ScratchTooSmall { needed, actual } => {
                write!(f, "{actual} scratch cells; the program needs {needed}")
            }
            Error::OutputLength { expected, actual } => {
                write!(f, "{actual} output cells for a program of {expected}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// FNV-1a over `bytes`, the format's checksum.
pub fn checksum(bytes: &[u8]) -> u32 {
    bytes.iter().fold(0x811C_9DC5u32, |hash, &byte| (hash ^ byte as u32).wrapping_mul(0x0100_0193))
}

/// Reads the format's little-endian words, refusing to read past the end.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, length: usize) -> Result<&'a [u8], Error> {
        let end = self.at.checked_add(length).ok_or(Error::Truncated)?;
        let taken = self.bytes.get(self.at..end).ok_or(Error::Truncated)?;
        self.at = end;
        Ok(taken)
    }

    fn u32(&mut self) -> Result<u32, Error> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn count(&mut self) -> Result<usize, Error> {
        Ok(self.u32()? as usize)
    }

    /// The offset of `count` words, skipped.
    fn words(&mut self, count: usize) -> Result<usize, Error> {
        let at = self.at;
        self.take(count.checked_mul(4).ok_or(Error::Truncated)?)?;
        Ok(at)
    }
}

fn word(bytes: &[u8], at: usize) -> u32 {
    let mut raw = [0u8; 4];
    raw.copy_from_slice(&bytes[at..at + 4]);
    u32::from_le_bytes(raw)
}

fn word_at(bytes: &[u8], base: usize, index: usize) -> u32 {
    word(bytes, base + 4 * index)
}

fn float_at(bytes: &[u8], base: usize, index: usize) -> f32 {
    f32::from_bits(word_at(bytes, base, index))
}

fn f64_at(bytes: &[u8], at: usize) -> f64 {
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&bytes[at..at + 8]);
    f64::from_le_bytes(raw)
}

/// Where one layer's parts are.
#[derive(Clone, Copy)]
struct Layer {
    output_size: usize,
    rows: usize,
    entries: usize,
    bias: usize,
    offsets: usize,
    positions: usize,
    weights: usize,
    gates: (usize, usize),
    booleans: (usize, usize),
    computes: (usize, usize),
    gather_reads: (usize, usize),
    index_reads: (usize, usize),
}

impl Layer {
    /// The layer at the reader, which it moves past. Only the sizes are read here; [`check`] checks the contents.
    fn read(reader: &mut Reader) -> Result<Layer, Error> {
        let output_size = reader.count()?;
        let rows = reader.count()?;
        let entries = reader.count()?;
        let gate_count = reader.count()?;
        let boolean_count = reader.count()?;
        let compute_count = reader.count()?;
        let gather_read_count = reader.count()?;
        let index_read_count = reader.count()?;
        let bias = reader.words(rows)?;
        let offsets = reader.words(rows.checked_add(1).ok_or(Error::Truncated)?)?;
        let positions = reader.words(entries)?;
        let weights = reader.words(entries)?;
        let gates = (reader.words(gate_count)?, gate_count);
        let booleans = (reader.words(boolean_count)?, boolean_count);
        let computes = (reader.at, compute_count);
        for _ in 0..compute_count {
            reader.u32()?;
            let count = reader.count()?;
            reader.words(count)?;
        }
        let gather_reads = (reader.at, gather_read_count);
        for _ in 0..gather_read_count {
            reader.words(3)?;
            let count = reader.count()?;
            reader.words(count)?;
        }
        let index_reads = (reader.at, index_read_count);
        for _ in 0..index_read_count {
            reader.words(3)?;
            let count = reader.count()?;
            reader.words(count)?;
        }
        Ok(Layer {
            output_size,
            rows,
            entries,
            bias,
            offsets,
            positions,
            weights,
            gates,
            booleans,
            computes,
            gather_reads,
            index_reads,
        })
    }
}

/// A check of one layer against the values before it: everything [`Program::run`] indexes is in bounds.
fn check(bytes: &[u8], layer: &Layer, available: usize, gather_slots: usize) -> Result<(), Error> {
    let fail = Error::Malformed;
    if layer.output_size.checked_add(layer.gates.1).map_or(true, |needed| needed > layer.rows) {
        return Err(fail("a layer has fewer rows than its outputs and gates"));
    }
    let entries = layer.entries;
    let mut previous = 0;
    for row in 0..=layer.rows {
        let offset = word_at(bytes, layer.offsets, row) as usize;
        if offset < previous || offset > entries {
            return Err(fail("a layer's row offsets aren't in order within its entries"));
        }
        previous = offset;
    }
    if previous != entries {
        return Err(fail("a layer's row offsets don't end at its entries"));
    }
    for entry in 0..entries {
        if word_at(bytes, layer.positions, entry) as usize >= available {
            return Err(fail("a layer reads a value computed after it"));
        }
    }
    for (base, count) in [layer.gates, layer.booleans] {
        for index in 0..count {
            if word_at(bytes, base, index) as usize >= layer.rows {
                return Err(fail("a gate or boolean names a row the layer doesn't have"));
            }
        }
    }
    let positions = |at: usize, count: usize| -> Result<(), Error> {
        for index in 0..count {
            if word_at(bytes, at, index) as usize >= available {
                return Err(fail("a read names a value computed after it"));
            }
        }
        Ok(())
    };
    let mut at = layer.computes.0;
    for _ in 0..layer.computes.1 {
        if word(bytes, at) as usize >= gather_slots {
            return Err(fail("a gather names a slot the program doesn't have"));
        }
        let count = word(bytes, at + 4) as usize;
        positions(at + 8, count)?;
        at += 8 + 4 * count;
    }
    let mut at = layer.gather_reads.0;
    for _ in 0..layer.gather_reads.1 {
        let (row, slot, masked) = (word(bytes, at) as usize, word(bytes, at + 4) as usize, word(bytes, at + 8));
        let count = word(bytes, at + 12) as usize;
        if row >= layer.rows || slot >= gather_slots || masked > 1 || count == 0 {
            return Err(fail("a gather read names a row or slot the layer doesn't have, or no candidates"));
        }
        positions(at + 16, count)?;
        at += 16 + 4 * count;
    }
    let mut at = layer.index_reads.0;
    for _ in 0..layer.index_reads.1 {
        let (row, index_row, flags) = (word(bytes, at) as usize, word(bytes, at + 4) as usize, word(bytes, at + 8));
        let count = word(bytes, at + 12) as usize;
        if row >= layer.rows || index_row >= layer.rows || flags > 3 || count == 0 || count > 1 << 24 {
            return Err(fail("an index read names a row the layer doesn't have, or no candidates"));
        }
        positions(at + 16, count)?;
        at += 16 + 4 * count;
    }
    Ok(())
}

/// A program checked by [`Program::parse`], borrowing its bytes.
#[derive(Clone, Copy, Debug)]
pub struct Program<'a> {
    bytes: &'a [u8],
    description: &'a str,
    source_size: usize,
    value_count: usize,
    gather_slots: usize,
    root_count: usize,
    layer_count: usize,
    sources: usize,
    /// Bytes per source cell: its range, and in version 2 its grid.
    source_bytes: usize,
    roots: usize,
    layers: usize,
    max_rows: usize,
}

impl<'a> Program<'a> {
    /// Check a program's bytes: the magic, the version, the checksum, and that every operation reads only values
    /// computed before it, so [`Program::run`] stays in bounds.
    pub fn parse(bytes: &'a [u8]) -> Result<Program<'a>, Error> {
        if bytes.len() < MAGIC.len() {
            return Err(if MAGIC.starts_with(bytes) { Error::Truncated } else { Error::NotAProgram });
        }
        if bytes[..MAGIC.len()] != MAGIC {
            return Err(Error::NotAProgram);
        }
        let mut reader = Reader { bytes, at: MAGIC.len() };
        let version = reader.u32()?;
        if version != 1 && version != FORMAT_VERSION {
            return Err(Error::UnsupportedVersion(version));
        }
        let source_bytes = if version == 1 { 16 } else { 24 };
        let Some(body_length) = bytes.len().checked_sub(4).filter(|&length| length >= reader.at) else {
            return Err(Error::Truncated);
        };
        if checksum(&bytes[..body_length]) != word(bytes, body_length) {
            return Err(Error::Checksum);
        }
        let mut reader = Reader { bytes: &bytes[..body_length], at: reader.at };
        let description_length = reader.count()?;
        let description = core::str::from_utf8(reader.take(description_length)?)
            .map_err(|_| Error::Malformed("the description isn't UTF-8"))?;
        let source_size = reader.count()?;
        let value_count = reader.count()?;
        let gather_slots = reader.count()?;
        let root_count = reader.count()?;
        let layer_count = reader.count()?;
        let sources = reader.words(source_size.checked_mul(source_bytes / 4).ok_or(Error::Truncated)?)?;
        if source_bytes == 24 {
            for index in 0..source_size {
                let step = f64_at(bytes, sources + 24 * index + 16);
                if step != 0.0 && grid_exponent(step).is_none() {
                    return Err(Error::Malformed("a grid step isn't 0 or a power of two from 2^-149 to 2^127"));
                }
            }
        }
        let roots = reader.words(root_count)?;
        for index in 0..root_count {
            if word_at(bytes, roots, index) as usize >= value_count {
                return Err(Error::Malformed("an output names a value the program doesn't have"));
            }
        }
        if source_size > value_count {
            return Err(Error::Malformed("more sources than values"));
        }
        let layers = reader.at;
        let mut available = source_size;
        let mut max_rows = 0;
        for _ in 0..layer_count {
            let layer = Layer::read(&mut reader)?;
            check(bytes, &layer, available, gather_slots)?;
            available = available
                .checked_add(layer.output_size)
                .filter(|&count| count <= value_count)
                .ok_or(Error::Malformed("the layers make more values than the program has"))?;
            max_rows = max_rows.max(layer.rows);
        }
        if available != value_count {
            return Err(Error::Malformed("the layers make fewer values than the program has"));
        }
        if reader.at != body_length {
            return Err(Error::Malformed("bytes after the last layer"));
        }
        // The scratch length must fit in a usize.
        value_count
            .checked_add(max_rows)
            .and_then(|length| length.checked_add(gather_slots.checked_mul(2)?))
            .ok_or(Error::Malformed("the program needs more scratch than memory holds"))?;
        Ok(Program {
            bytes: &bytes[..body_length],
            description,
            source_size,
            value_count,
            gather_slots,
            root_count,
            layer_count,
            sources,
            source_bytes,
            roots,
            layers,
            max_rows,
        })
    }

    /// What the exporter wrote about the program, e.g. its function's name and signature.
    pub fn description(&self) -> &'a str {
        self.description
    }

    /// The number of input cells, the parameters' cells in order.
    pub fn input_count(&self) -> usize {
        self.source_size
    }

    /// The number of output cells, the result's cells in order.
    pub fn output_count(&self) -> usize {
        self.root_count
    }

    /// The number of layers, each a dependent step: the program's depth.
    pub fn layer_count(&self) -> usize {
        self.layer_count
    }

    /// The number of values a run computes, the input cells included.
    pub fn value_count(&self) -> usize {
        self.value_count
    }

    /// The range declared for input cell `index`, inclusive.
    pub fn input_range(&self, index: usize) -> Option<(f64, f64)> {
        (index < self.source_size).then(|| {
            let at = self.sources + self.source_bytes * index;
            (f64_at(self.bytes, at), f64_at(self.bytes, at + 8))
        })
    }

    /// The step input cell `index`'s values are whole multiples of, e.g. 1 for an integer; `None` where any f32 in
    /// its range is allowed, and for every cell of a version 1 program, which carries no grids.
    pub fn input_grid(&self, index: usize) -> Option<f64> {
        if index >= self.source_size || self.source_bytes < 24 {
            return None;
        }
        Some(f64_at(self.bytes, self.sources + 24 * index + 16)).filter(|&step| step != 0.0)
    }

    /// The f32 cells of scratch memory [`Program::run`] needs.
    pub fn scratch_len(&self) -> usize {
        self.value_count + self.max_rows + 2 * self.gather_slots
    }

    /// Evaluate the program on `inputs`, writing its outputs. `scratch` holds at least [`Program::scratch_len`]
    /// cells; afterwards its first [`Program::value_count`] cells hold every value the run computed, in program
    /// order. Inputs outside their declared ranges or off their grids are refused.
    pub fn run(&self, inputs: &[f32], scratch: &mut [f32], outputs: &mut [f32]) -> Result<(), Error> {
        if inputs.len() != self.source_size {
            return Err(Error::InputLength { expected: self.source_size, actual: inputs.len() });
        }
        if scratch.len() < self.scratch_len() {
            return Err(Error::ScratchTooSmall { needed: self.scratch_len(), actual: scratch.len() });
        }
        if outputs.len() != self.root_count {
            return Err(Error::OutputLength { expected: self.root_count, actual: outputs.len() });
        }
        for (index, &value) in inputs.iter().enumerate() {
            // The same test as `value as f64` within the f64 bounds, but on f32: the bounds' conversion depends only
            // on the program, so the time doesn't depend on the inputs on chips whose FPU lacks f64.
            let (low, high) = self.input_range(index).unwrap_or((f64::NAN, f64::NAN));
            if !(value >= f32_at_least(low) && value <= f32_at_most(high)) {
                return Err(Error::InputOutOfRange { index, value });
            }
            if let Some(step) = self.input_grid(index) {
                // parse checked that every step has an exponent.
                if !on_grid(value, grid_exponent(step).unwrap_or(0)) {
                    return Err(Error::InputOffGrid { index, value, step });
                }
            }
        }
        let bytes = self.bytes;
        let (values, rest) = scratch.split_at_mut(self.value_count);
        let (projection, rest) = rest.split_at_mut(self.max_rows);
        let (gather_indices, rest) = rest.split_at_mut(self.gather_slots);
        let gather_masses = &mut rest[..self.gather_slots];
        gather_indices.fill(0.0);
        gather_masses.fill(0.0);
        values[..inputs.len()].copy_from_slice(inputs);
        let mut available = inputs.len();
        let mut reader = Reader { bytes, at: self.layers };
        for _ in 0..self.layer_count {
            let Ok(layer) = Layer::read(&mut reader) else { unreachable!("parse checked every layer") };
            let projection = &mut projection[..layer.rows];
            // One sum per row, its entries in stored order.
            for (row, output) in projection.iter_mut().enumerate() {
                let mut value = float_at(bytes, layer.bias, row);
                let (start, end) = (word_at(bytes, layer.offsets, row), word_at(bytes, layer.offsets, row + 1));
                for entry in start as usize..end as usize {
                    let input = values[word_at(bytes, layer.positions, entry) as usize];
                    value += float_at(bytes, layer.weights, entry) * input;
                }
                *output = value;
            }
            // A one-hot group's selected position and its sum of bits; the position is kept as an i32's bits.
            let mut at = layer.computes.0;
            for _ in 0..layer.computes.1 {
                let slot = word(bytes, at) as usize;
                let count = word(bytes, at + 4) as usize;
                let (mut index, mut mass) = (0f64, 0f64);
                for position in 0..count {
                    let bit = values[word_at(bytes, at + 8, position) as usize] as f64;
                    index += position as f64 * bit;
                    mass += bit;
                }
                gather_indices[slot] = f32::from_bits(index as i32 as u32);
                gather_masses[slot] = mass as f32;
                at += 8 + 4 * count;
            }
            let mut at = layer.gather_reads.0;
            for _ in 0..layer.gather_reads.1 {
                let (row, slot, masked) = (word(bytes, at) as usize, word(bytes, at + 4) as usize, word(bytes, at + 8));
                let count = word(bytes, at + 12) as usize;
                let index = (gather_indices[slot].to_bits() as i32).clamp(0, count as i32 - 1) as usize;
                let candidate = values[word_at(bytes, at + 16, index) as usize];
                projection[row] += if masked == 1 { gather_masses[slot] * candidate } else { candidate };
                at += 16 + 4 * count;
            }
            // A table read at a row's value: the floor of the index picks the candidate; outside the table, 0 or
            // the nearest end.
            let mut at = layer.index_reads.0;
            for _ in 0..layer.index_reads.1 {
                let (row, index_row, flags) =
                    (word(bytes, at) as usize, word(bytes, at + 4) as usize, word(bytes, at + 8));
                let count = word(bytes, at + 12) as usize;
                // Without branches on the index: the candidate at the clamped position, masked to 0 where the index
                // is past an end whose rule is 0. Comparing in f32 is exact, since tables hold at most 2^24 entries.
                let index = projection[index_row];
                let below = index < 0.0;
                let above = index >= count as f32;
                let position = (index as i32).clamp(0, count as i32 - 1) as usize;
                let zero = (below & (flags & 1 != 0)) | (above & (flags & 2 != 0));
                let candidate = values[word_at(bytes, at + 16, position) as usize];
                projection[row] = if index.is_nan() { f32::NAN } else { keep_if(!zero, candidate) };
                at += 16 + 4 * count;
            }
            for gate in 0..layer.gates.1 {
                let row = word_at(bytes, layer.gates.0, gate) as usize;
                projection[row] *= relu(projection[layer.output_size + gate]);
            }
            for boolean in 0..layer.booleans.1 {
                let row = word_at(bytes, layer.booleans.0, boolean) as usize;
                projection[row] = round_boolean(projection[row]);
            }
            values[available..available + layer.output_size].copy_from_slice(&projection[..layer.output_size]);
            available += layer.output_size;
        }
        for (index, output) in outputs.iter_mut().enumerate() {
            *output = values[word_at(bytes, self.roots, index) as usize];
        }
        Ok(())
    }

    /// [`Program::run`] with scratch and outputs of its own.
    #[cfg(feature = "alloc")]
    pub fn evaluate(&self, inputs: &[f32]) -> Result<alloc::vec::Vec<f32>, Error> {
        let mut scratch = alloc::vec![0.0; self.scratch_len()];
        let mut outputs = alloc::vec![0.0; self.root_count];
        self.run(inputs, &mut scratch, &mut outputs)?;
        Ok(outputs)
    }
}

/// The least f32 at or above `bound`: an f32 is at least `bound` exactly when it's at least this.
fn f32_at_least(bound: f64) -> f32 {
    let near = bound as f32;
    if (near as f64) < bound {
        next_f32(near, true)
    } else {
        near
    }
}

/// The greatest f32 at or below `bound`.
fn f32_at_most(bound: f64) -> f32 {
    let near = bound as f32;
    if (near as f64) > bound {
        next_f32(near, false)
    } else {
        near
    }
}

/// The next f32 up or down from a finite `value`.
fn next_f32(value: f32, up: bool) -> f32 {
    if value == 0.0 {
        let tiny = f32::from_bits(1);
        return if up { tiny } else { -tiny };
    }
    let bits = value.to_bits();
    let away = (value > 0.0) == up;
    f32::from_bits(if away { bits + 1 } else { bits - 1 })
}

/// `k` where `step` is 2^k, a grid step the format allows: 2^-149 to 2^127.
fn grid_exponent(step: f64) -> Option<i32> {
    let bits = step.to_bits();
    let exponent = ((bits >> 52) & 0x7FF) as i32 - 1023;
    let power_of_two = bits >> 63 == 0 && bits & ((1 << 52) - 1) == 0;
    (power_of_two && (-149..=127).contains(&exponent)).then_some(exponent)
}

/// Whether a finite `value` is a whole multiple of 2^`exponent`, from its bits alone, so neither the time nor the
/// result depends on how a chip rounds: no float operations and no branches on the value.
fn on_grid(value: f32, exponent: i32) -> bool {
    let bits = value.to_bits();
    let field = ((bits >> 23) & 0xFF) as i32;
    let normal = field != 0;
    // value = ±significand × 2^scale with an integer significand; subnormals have scale -149 like the smallest
    // normals.
    let significand = (bits & 0x7F_FFFF) | ((normal as u32) << 23);
    let scale = field - 150 + !normal as i32;
    // The significand's bits worth less than 2^exponent must be zero.
    let low_bits = (exponent - scale).clamp(0, 32) as u32;
    // (2^low_bits) - 1 in 32-bit shifts, which chips without 64-bit ones do in fixed time: all ones at 32.
    let mask = (1u32 << (low_bits & 31)).wrapping_sub(1) | 0u32.wrapping_sub(low_bits >> 5);
    significand & mask == 0
}

/// `max(0, value)`, keeping NaN.
fn relu(value: f32) -> f32 {
    if value.is_nan() {
        return f32::NAN;
    }
    // A mask rather than a branch on the sign, so a run's time doesn't depend on the values: the value where it's
    // positive, else +0.
    keep_if(value > 0.0, value)
}

/// `value` where `keep`, else +0, without a branch.
fn keep_if(keep: bool, value: f32) -> f32 {
    f32::from_bits(value.to_bits() & 0u32.wrapping_sub(keep as u32))
}

/// A score within 0..1 rounded to 0 below 0.5 and 1 from 0.5: `(score + 8,388,609) - 8,388,609` in f32.
fn round_boolean(score: f32) -> f32 {
    const BASE: f32 = 8_388_609.0;
    (score + BASE) - BASE
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bit test agrees with dividing by the step in f64, which is exact for these values and steps.
    #[test]
    fn grid_test_matches_division() {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };
        let specials = [0.0, -0.0, 0.5, -0.5, 1.0, 1.5, 3.0, -7.0, 0.25, 16_777_216.0, 16_777_215.0, 1e-45, 3.4e38];
        for round in 0..300_000 {
            let value = if round < specials.len() { specials[round] } else { f32::from_bits(next() as u32) };
            if !value.is_finite() {
                continue;
            }
            for exponent in [-149, -126, -20, -2, -1, 0, 1, 3, 23, 24, 30, 127] {
                let scaled = value as f64 / power_of_two(exponent);
                // Every f64 of magnitude 2^53 or more is whole.
                let whole = !(-9e15..=9e15).contains(&scaled) || scaled == (scaled as i64) as f64;
                assert_eq!(on_grid(value, exponent), whole, "{value} on 2^{exponent}");
            }
        }
        assert!(on_grid(20.0, 0) && !on_grid(19.5, 0) && on_grid(19.5, -1) && !on_grid(6.0, 2));
    }

    /// 2^`exponent` for exponents an f64 holds as a normal number.
    fn power_of_two(exponent: i32) -> f64 {
        f64::from_bits(((exponent + 1023) as u64) << 52)
    }

    #[test]
    fn grid_steps_are_powers_of_two() {
        assert_eq!(grid_exponent(1.0), Some(0));
        assert_eq!(grid_exponent(0.25), Some(-2));
        assert_eq!(grid_exponent(power_of_two(-149)), Some(-149));
        assert_eq!(grid_exponent(power_of_two(127)), Some(127));
        for step in [0.0, -1.0, 3.0, 0.3, power_of_two(-150), power_of_two(128), f64::INFINITY, f64::NAN] {
            assert_eq!(grid_exponent(step), None, "{step}");
        }
    }

    /// The f32 range test accepts exactly the values the f64 test does.
    #[test]
    fn f32_bounds_match_f64_comparisons() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };
        let specials = [0.0, -0.0, 1e-40, -1e-40, 0.1, -0.1, 16_777_217.0, 3.4e38, 1e300, -1e300, f64::INFINITY];
        for round in 0..200_000 {
            let bound = if round < specials.len() * 8 {
                specials[round % specials.len()]
            } else {
                f64::from_bits(next()) % 1e6
            };
            if bound.is_nan() {
                continue;
            }
            let near = bound as f32;
            let candidates = [near, next_f32(near, true), next_f32(near, false), f32::from_bits(next() as u32)];
            for value in candidates {
                if value.is_nan() || !near.is_finite() && value == near {
                    continue;
                }
                assert_eq!(value >= f32_at_least(bound), value as f64 >= bound, "{value} >= {bound}");
                assert_eq!(value <= f32_at_most(bound), value as f64 <= bound, "{value} <= {bound}");
            }
        }
    }
}
