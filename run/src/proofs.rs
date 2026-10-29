//! Kani proofs of the runner (`run/kani.sh`). Compiled only by Kani: the runner itself doesn't change.
//!
//! - Every numeric helper does what its comment says, for every f32 or f64 it can be given.
//! - `Program::parse` doesn't panic, in two parts: `a_layer_never_panics`, reading and checking one layer of any
//!   `LAYER` bytes, and `parse_never_panics`, everything else on any file of at most `FILE` bytes, with a layer read
//!   and checked by stand-ins that give any answer the real ones could.
//! - `Program::run` doesn't panic on any program of a shape with every kind of operation that `parse` accepts, on
//!   any inputs: the shape fixes the counts, and every other word of the file is Kani's. With no `unsafe` in the
//!   crate, that is what "a run can't read out of bounds" means. Wrong lengths of inputs, scratch and outputs get
//!   their errors.
//!
//! The checksum is replaced by a constant, so the proofs cover every body as if its checksum matched, and the
//! description's UTF-8 check by a stand-in that gives either answer. Kani proves them for the host's 64-bit
//! `usize`, and each `kani::cover!` shows its proof reaches the case it's about.

#![allow(dead_code)] // the stand-ins are named only in kani::stub attributes

use super::*;

/// 2^k as an f64, exactly, for k in -1022..=1023.
fn power_of_two(k: i32) -> f64 {
    f64::from_bits(((k + 1023) as u64) << 52)
}

#[kani::proof]
fn grid_exponent_finds_exactly_the_allowed_powers_of_two() {
    let step: f64 = kani::any();
    match grid_exponent(step) {
        Some(k) => assert!((-149..=127).contains(&k) && step.to_bits() == power_of_two(k).to_bits()),
        None => {
            let k: i32 = kani::any();
            kani::assume((-149..=127).contains(&k));
            assert!(step.to_bits() != power_of_two(k).to_bits());
        }
    }
}

/// The grid test is divisibility: a finite value is on the grid of step 2^k exactly when value / 2^k is a whole
/// number. Dividing by 2^k is multiplying by 2^-k, exact in f64 for every f32 and allowed k, and an f64 of magnitude
/// 2^53 or more is whole.
#[kani::proof]
fn on_grid_is_divisibility() {
    let value: f32 = kani::any();
    kani::assume(value.is_finite());
    let k: i32 = kani::any();
    kani::assume((-149..=127).contains(&k));
    let scaled = value as f64 * power_of_two(-k);
    let whole = scaled.abs() >= 9_007_199_254_740_992.0 || scaled == (scaled as i64) as f64;
    assert_eq!(on_grid(value, k), whole);
}

/// The range check in f32 is the comparison in f64, for every f32 and every bound, NaN and infinities included.
#[kani::proof]
fn f32_at_least_is_the_f64_comparison() {
    let bound: f64 = kani::any();
    let value: f32 = kani::any();
    assert_eq!(value >= f32_at_least(bound), value as f64 >= bound);
}

#[kani::proof]
fn f32_at_most_is_the_f64_comparison() {
    let bound: f64 = kani::any();
    let value: f32 = kani::any();
    assert_eq!(value <= f32_at_most(bound), value as f64 <= bound);
}

#[kani::proof]
fn next_f32_is_the_neighbour() {
    let value: f32 = kani::any();
    kani::assume(value.is_finite());
    let up: bool = kani::any();
    let next = next_f32(value, up);
    let between: f32 = kani::any();
    if up {
        assert!(next > value && !(value < between && between < next));
    } else {
        assert!(next < value && !(next < between && between < value));
    }
}

#[kani::proof]
fn relu_and_keep_if_are_their_definitions() {
    let value: f32 = kani::any();
    let result = relu(value);
    if value.is_nan() {
        assert!(result.is_nan());
    } else if value > 0.0 {
        assert_eq!(result.to_bits(), value.to_bits());
    } else {
        assert_eq!(result.to_bits(), 0); // +0, also for -0 and every negative value
    }
    let keep: bool = kani::any();
    assert_eq!(keep_if(keep, value).to_bits(), if keep { value.to_bits() } else { 0 });
}

#[kani::proof]
fn round_boolean_rounds_half_up() {
    let score: f32 = kani::any();
    kani::assume((0.0..=1.0).contains(&score));
    let result = round_boolean(score);
    assert_eq!(result.to_bits(), if score < 0.5 { 0.0f32 } else { 1.0f32 }.to_bits());
}

fn constant_checksum(_bytes: &[u8]) -> u32 {
    0
}

/// UTF-8 or not, chosen by Kani: nothing after `parse` reads the description, so this covers every outcome.
#[allow(invalid_from_utf8)]
fn any_utf8(_bytes: &[u8]) -> Result<&str, core::str::Utf8Error> {
    if kani::any() {
        Ok("")
    } else {
        core::str::from_utf8_mut(&mut [0xFF]).map(|_| "") // not from_utf8, which this stands in for
    }
}

/// Any layer `Layer::read` could return, or its error: on success it has read at least the eight counts and one row
/// offset, 36 bytes.
fn any_layer(reader: &mut Reader) -> Result<Layer, Error> {
    if kani::any() {
        return Err(Error::Truncated);
    }
    let end: usize = kani::any();
    kani::assume(end >= reader.at + 36 && end <= reader.bytes.len());
    reader.at = end;
    Ok(Layer {
        output_size: kani::any(),
        rows: kani::any(),
        entries: kani::any(),
        bias: kani::any(),
        offsets: kani::any(),
        positions: kani::any(),
        weights: kani::any(),
        gates: kani::any(),
        booleans: kani::any(),
        computes: kani::any(),
        gather_reads: kani::any(),
        index_reads: kani::any(),
    })
}

/// Either of `check`'s answers.
fn any_check(_bytes: &[u8], _layer: &Layer, _available: usize, _gather_slots: usize) -> Result<(), Error> {
    if kani::any() {
        Ok(())
    } else {
        Err(Error::Malformed("any"))
    }
}

/// The longest file the parse proof covers, in bytes: room for the header and a layer.
const FILE: usize = 96;

/// Every file of at most `FILE` bytes: `parse` returns, and every accessor of a program it accepts returns. Each
/// layer is read and checked by stand-ins that give any answer the real ones could, so this covers everything parse
/// does but those two, which `a_layer_never_panics` covers.
#[kani::proof]
#[kani::stub(checksum, constant_checksum)]
#[kani::stub(core::str::from_utf8, any_utf8)]
#[kani::stub(Layer::read, any_layer)]
#[kani::stub(check, any_check)]
#[kani::unwind(16)]
fn parse_never_panics() {
    let bytes: [u8; FILE] = kani::any();
    let length: usize = kani::any();
    kani::assume(length <= FILE);
    let Ok(program) = Program::parse(&bytes[..length]) else { return };
    kani::cover!(program.layer_count() == 1, "a file with a layer is accepted");
    let _ = (program.description(), program.input_count(), program.output_count(), program.scratch_len());
    let index: usize = kani::any();
    let _ = (program.input_range(index), program.input_grid(index), program.input_variant(index));
    if let Some(values) = program.input_values(index) {
        for value in values {
            let _ = value;
        }
    };
}

/// The longest layer `a_layer_never_panics` covers, in bytes.
const LAYER: usize = 64;

/// Every layer of at most `LAYER` bytes is read and checked without a panic, whatever came before it.
#[kani::proof]
#[kani::unwind(9)]
fn a_layer_never_panics() {
    let bytes: [u8; LAYER] = kani::any();
    let length: usize = kani::any();
    kani::assume(length <= LAYER);
    let bytes = &bytes[..length];
    let mut reader = Reader { bytes, at: 0 };
    if let Ok(layer) = Layer::read(&mut reader) {
        let checked = check(bytes, &layer, kani::any(), kani::any());
        kani::cover!(checked.is_ok() && layer.entries > 0, "a layer with entries passes");
    }
}

/// The counts of a program, which fix where everything in its file is; the words they don't fix are Kani's.
struct Shape {
    version: u32,
    inputs: usize,
    /// The number of values in each value set.
    value_sets: &'static [usize],
    outputs: usize,
    layers: &'static [LayerShape],
}

struct LayerShape {
    rows: usize,
    entries: usize,
    gates: usize,
    booleans: usize,
    /// The number of candidates of each one-hot group, gather read and index read.
    computes: &'static [usize],
    gather_reads: &'static [usize],
    index_reads: &'static [usize],
}

/// The longest file a shape may have, in bytes.
const SHAPED: usize = 512;

/// Writes a file's words in order. The builder uses index loops only: Kani loses track of the place of some
/// iterators (a Chain, or one over an array of slices), and then unwinds the loops they drive to the bound.
struct Writer<'a> {
    bytes: &'a mut [u8; SHAPED],
    at: usize,
}

impl Writer<'_> {
    fn put(&mut self, word: u32) {
        self.bytes[self.at..self.at + 4].copy_from_slice(&word.to_le_bytes());
        self.at += 4;
    }

    fn any(&mut self, count: usize) {
        let mut index = 0;
        while index < count {
            self.put(kani::any());
            index += 1;
        }
    }

    /// Reads of `counts.len()` kinds, each `head` words chosen by Kani, its count, and that many positions.
    fn reads(&mut self, head: usize, counts: &[usize]) {
        let mut index = 0;
        while index < counts.len() {
            self.any(head);
            self.put(counts[index] as u32);
            self.any(counts[index]);
            index += 1;
        }
    }
}

/// The file of a program of `shape`, and the program `parse` must make of it if it accepts it. Every count is the
/// shape's and every other word Kani's (the checksum is stubbed), so the programs `parse` accepts of these files are
/// every program of that shape.
fn file_of_shape<'a>(shape: &Shape, bytes: &'a mut [u8; SHAPED]) -> (&'a [u8], Program<'static>) {
    let mut file = Writer { bytes, at: 0 };
    file.put(u32::from_le_bytes(*b"HWCI"));
    file.put(u32::from_le_bytes(*b"RC\0\0"));
    file.put(shape.version);
    file.put(0); // the description, which parse_never_panics covers
    file.put(shape.inputs as u32);
    file.any(2); // values, gather slots
    file.put(shape.outputs as u32);
    file.put(shape.layers.len() as u32);
    if shape.version >= 3 {
        file.put(shape.value_sets.len() as u32);
    }
    let source_bytes = if shape.version == 1 {
        16
    } else if shape.version == 2 {
        24
    } else {
        32
    };
    let sources = file.at;
    file.any(shape.inputs * source_bytes / 4);
    let value_sets = (file.at, shape.value_sets.len());
    file.reads(1, shape.value_sets); // each its cell, count and values
    let roots = file.at;
    file.any(shape.outputs);
    let layers = file.at;
    let (mut index, mut max_rows) = (0, 0);
    while index < shape.layers.len() {
        let layer = &shape.layers[index];
        max_rows = max_rows.max(layer.rows);
        file.any(1); // output size
        file.put(layer.rows as u32);
        file.put(layer.entries as u32);
        file.put(layer.gates as u32);
        file.put(layer.booleans as u32);
        file.put(layer.computes.len() as u32);
        file.put(layer.gather_reads.len() as u32);
        file.put(layer.index_reads.len() as u32);
        // Biases, row offsets, positions, weights, and the rows of gates and booleans.
        file.any(layer.rows + (layer.rows + 1) + 2 * layer.entries + layer.gates + layer.booleans);
        file.reads(1, layer.computes); // its slot
        file.reads(3, layer.gather_reads); // its row, slot and mask
        file.reads(3, layer.index_reads); // its row, index row and flags
        index += 1;
    }
    file.put(0); // the checksum, which constant_checksum gives
    let length = file.at;
    let bytes = &bytes[..length];
    // Its bytes are the caller's to set: they're borrowed from the file this returns.
    let program = Program {
        bytes: &[],
        description: "",
        source_size: shape.inputs,
        value_count: word(bytes, 20) as usize,
        gather_slots: word(bytes, 24) as usize,
        root_count: shape.outputs,
        layer_count: shape.layers.len(),
        sources,
        source_bytes,
        value_sets,
        roots,
        layers,
        max_rows,
    };
    (bytes, program)
}

/// Cells of scratch the run proofs offer; a program that needs more gets an error, never a panic.
const CELLS: usize = 16;

/// The program `parse` makes of a file of `shape`, and whether it accepts it: the shape's layout, checked to be
/// what parse found. A run needs the layout as plain numbers, and Kani merges what a function returns over its
/// every path, so the program is made before parse is asked, outside any branch, and none of parse's refusals reach
/// it.
fn parse_shape<'a>(shape: &Shape, bytes: &'a mut [u8; SHAPED]) -> (Program<'a>, bool) {
    let (bytes, expected) = file_of_shape(shape, bytes);
    let program = Program { bytes: &bytes[..bytes.len() - 4], ..expected };
    let parsed = Program::parse(bytes);
    if let Ok(parsed) = parsed {
        assert!(parsed.bytes.len() == program.bytes.len() && parsed.description.is_empty());
        assert!(
            (parsed.source_size, parsed.value_count, parsed.gather_slots)
                == (program.source_size, program.value_count, program.gather_slots)
        );
        assert!(
            (parsed.root_count, parsed.layer_count, parsed.sources)
                == (program.root_count, program.layer_count, program.sources)
        );
        assert!(
            (parsed.source_bytes, parsed.value_sets, parsed.roots)
                == (program.source_bytes, program.value_sets, program.roots)
        );
        assert!((parsed.layers, parsed.max_rows) == (program.layers, program.max_rows));
    }
    (program, parsed.is_ok())
}

/// Every program of `shape` that `parse` accepts runs without a panic on every input, with `CELLS` cells of scratch.
fn runs_without_panic(shape: &Shape) {
    let mut bytes = [0u8; SHAPED];
    let (program, accepted) = parse_shape(shape, &mut bytes);
    if !accepted {
        return;
    }
    let inputs: [f32; CELLS] = kani::any();
    let mut scratch = [0f32; CELLS];
    let mut outputs = [0f32; CELLS];
    let result = program.run(&inputs[..shape.inputs], &mut scratch, &mut outputs[..shape.outputs]);
    kani::cover!(result.is_ok(), "a run of the shape completes");
}

/// One layer with every kind of operation: a sum, a gate, a boolean, a one-hot group, a gather read and an index
/// read, on two inputs, one with a value set.
const EVERY_OPERATION: LayerShape =
    LayerShape { rows: 3, entries: 3, gates: 1, booleans: 1, computes: &[2], gather_reads: &[2], index_reads: &[2] };

#[kani::proof]
#[kani::stub(checksum, constant_checksum)]
#[kani::unwind(6)]
fn every_operation_runs_without_panic() {
    runs_without_panic(&Shape { version: 3, inputs: 2, value_sets: &[2], outputs: 2, layers: &[EVERY_OPERATION] });
}

/// The longest inputs, scratch and outputs `wrong_lengths_get_their_errors` gives.
const LENGTHS: usize = 4;

/// A run given inputs, scratch or outputs of a wrong length gets that error, checked before anything else: on a
/// program of two inputs and two outputs without layers, with every length up to `LENGTHS`.
#[kani::proof]
#[kani::stub(checksum, constant_checksum)]
#[kani::unwind(6)]
fn wrong_lengths_get_their_errors() {
    let mut bytes = [0u8; SHAPED];
    let shape = Shape { version: 3, inputs: 2, value_sets: &[], outputs: 2, layers: &[] };
    let (program, accepted) = parse_shape(&shape, &mut bytes);
    if !accepted {
        return;
    }
    let inputs: [f32; LENGTHS] = kani::any();
    let mut scratch = [0f32; LENGTHS];
    let mut outputs = [0f32; LENGTHS];
    let (given_inputs, given_scratch, given_outputs): (usize, usize, usize) = (kani::any(), kani::any(), kani::any());
    kani::assume(given_inputs <= LENGTHS && given_scratch <= LENGTHS && given_outputs <= LENGTHS);
    let result = program.run(&inputs[..given_inputs], &mut scratch[..given_scratch], &mut outputs[..given_outputs]);
    kani::cover!(result.is_ok(), "a run with the right lengths completes");
    if given_inputs != program.input_count() {
        assert!(matches!(result, Err(Error::InputLength { .. })));
    } else if given_scratch < program.scratch_len() {
        assert!(matches!(result, Err(Error::ScratchTooSmall { .. })));
    } else if given_outputs != program.output_count() {
        assert!(matches!(result, Err(Error::OutputLength { .. })));
    }
}
