//! A CPU executor for many records at once.
//!
//! Values are stored per wire as a block of [`LANES`] records, so every row of
//! a layer is a loop over records that the compiler can vectorize. Each lane
//! still computes `bias + w0 × x0 + w1 × x1 + ...` in the stored order with
//! FP32 rounding after every step (Rust never fuses or regroups float
//! arithmetic), so every value is bit for bit the one [`TensorExecutor`]
//! computes. [`BatchExecutor::run_parallel`] splits the records across threads.
//!
//! [`TensorExecutor`]: crate::TensorExecutor

use crate::circuit::{OutOfRange, resolve_index_position};
use crate::numeric::{evaluate_boolean, relu32};
use crate::tensor::{CandidatePositions, GatherCompute, GatherRead, TensorProgram, resolve_gather};

/// Records evaluated together.
pub const LANES: usize = 64;

type Block = [f32; LANES];

struct Layer {
    projection_size: usize,
    output_size: usize,
    bias: Vec<f32>,
    row_offsets: Vec<u32>,
    /// Packed value positions, read directly instead of through a gathered input.
    positions: Vec<u32>,
    weights: Vec<f32>,
    gate_rows: Vec<u32>,
    boolean_rows: Vec<u32>,
    gather_computes: Vec<GatherCompute>,
    gather_reads: Vec<GatherRead>,
    index_reads: Vec<IndexRead>,
}

struct IndexRead {
    row: usize,
    index_row: usize,
    below: OutOfRange,
    above: OutOfRange,
    candidate_positions: CandidatePositions,
}

/// Reusable buffers for running one program on batches of records.
pub struct BatchExecutor<'program> {
    program: &'program TensorProgram,
    layers: Vec<Layer>,
    root_positions: Vec<usize>,
    values: Vec<Block>,
    projection: Vec<Block>,
    gather_indices: Vec<[i32; LANES]>,
    gather_masses: Vec<Block>,
}

impl<'program> BatchExecutor<'program> {
    pub fn new(program: &'program TensorProgram) -> BatchExecutor<'program> {
        let layers: Vec<Layer> = program
            .layers
            .iter()
            .map(|layer| Layer {
                projection_size: layer.bias.len(),
                output_size: layer.output_size,
                bias: layer.bias.clone(),
                row_offsets: layer.row_offsets.clone(),
                positions: layer.columns.iter().map(|&column| layer.input_indices[column as usize]).collect(),
                weights: layer.weights.clone(),
                gate_rows: layer.gate_rows.clone(),
                boolean_rows: layer.boolean_rows.clone(),
                gather_computes: layer.gather_computes.clone(),
                gather_reads: layer.gather_reads.clone(),
                index_reads: layer
                    .index_reads
                    .iter()
                    .map(|read| IndexRead {
                        row: read.row as usize,
                        index_row: read.index_row as usize,
                        below: if read.below_zero { OutOfRange::Zero } else { OutOfRange::Clamp },
                        above: if read.above_zero { OutOfRange::Zero } else { OutOfRange::Clamp },
                        candidate_positions: read.candidate_positions.clone(),
                    })
                    .collect(),
            })
            .collect();
        let max_projection = layers.iter().map(|layer| layer.projection_size).max().unwrap_or(0);
        let root_positions = program
            .roots
            .iter()
            .map(|root| program.wire_order.iter().position(|wire| wire == root).expect("root was removed"))
            .collect();
        BatchExecutor {
            program,
            layers,
            root_positions,
            values: vec![[0.0; LANES]; program.value_count],
            projection: vec![[0.0; LANES]; max_projection],
            gather_indices: vec![[0; LANES]; program.gather_slot_count],
            gather_masses: vec![[0.0; LANES]; program.gather_slot_count],
        }
    }

    /// Evaluate the program on consecutive source tables, `program.source_size`
    /// values each, and return the roots' values, `program.roots.len()` per
    /// record. Sources aren't range-checked; see [`Circuit::validate_source_ranges`].
    ///
    /// [`Circuit::validate_source_ranges`]: crate::Circuit::validate_source_ranges
    pub fn run(&mut self, sources: &[f32]) -> Vec<f32> {
        let mut outputs = Vec::new();
        self.run_into(sources, &mut outputs);
        outputs
    }

    /// [`run`](Self::run), appending to `outputs`.
    pub fn run_into(&mut self, sources: &[f32], outputs: &mut Vec<f32>) {
        let source_size = self.program.source_size;
        let root_count = self.program.roots.len();
        let records = sources.len().checked_div(source_size).unwrap_or(0);
        assert_eq!(records * source_size, sources.len(), "sources aren't whole source tables");
        outputs.reserve(records * root_count);
        for first in (0..records).step_by(LANES) {
            let count = (records - first).min(LANES);
            for cell in 0..source_size {
                let block = &mut self.values[cell];
                for (lane, value) in block.iter_mut().enumerate() {
                    // A partial batch repeats its last record in the unused lanes.
                    *value = sources[(first + lane.min(count - 1)) * source_size + cell];
                }
            }
            self.evaluate();
            for lane in 0..count {
                outputs.extend(self.root_positions.iter().map(|&position| self.values[position][lane]));
            }
        }
    }

    fn evaluate(&mut self) {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU supports AVX2.
            return unsafe { self.evaluate_avx2() };
        }
        self.evaluate_portable();
    }

    /// The same code with 256-bit vectors. Only AVX2 is enabled, not FMA: Rust never contracts `a + w * x` into a
    /// fused multiply-add anyway, but this keeps it out of reach.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn evaluate_avx2(&mut self) {
        self.evaluate_portable();
    }

    // Lane loops index several blocks at once.
    #[allow(clippy::needless_range_loop)]
    #[inline(always)]
    fn evaluate_portable(&mut self) {
        let values = &mut self.values;
        let mut available = self.program.source_size;
        for layer in &self.layers {
            let projection = &mut self.projection[..layer.projection_size];
            for (row, block) in projection.iter_mut().enumerate() {
                *block = [layer.bias[row]; LANES];
                let entries = layer.row_offsets[row] as usize..layer.row_offsets[row + 1] as usize;
                for (&position, &weight) in layer.positions[entries.clone()].iter().zip(&layer.weights[entries]) {
                    let input = &values[position as usize];
                    for lane in 0..LANES {
                        block[lane] += weight * input[lane];
                    }
                }
            }
            for compute in &layer.gather_computes {
                let slot = compute.slot as usize;
                for lane in 0..LANES {
                    let bits = compute.bit_positions.iter().map(|&bit| values[bit as usize][lane]);
                    (self.gather_indices[slot][lane], self.gather_masses[slot][lane]) = resolve_gather(bits);
                }
            }
            for read in &layer.gather_reads {
                let slot = read.slot as usize;
                let limit = read.candidate_positions.len() as i32 - 1;
                for lane in 0..LANES {
                    // As in `TensorExecutor`: the clamp keeps a violated declaration inside the table.
                    let position = read.candidate_positions[self.gather_indices[slot][lane].clamp(0, limit) as usize];
                    let candidate = values[position as usize][lane];
                    let contribution = if read.masked { self.gather_masses[slot][lane] * candidate } else { candidate };
                    projection[read.row as usize][lane] += contribution;
                }
            }
            for read in &layer.index_reads {
                for lane in 0..LANES {
                    let index = projection[read.index_row][lane] as f64;
                    let count = read.candidate_positions.len();
                    projection[read.row][lane] = match resolve_index_position(index, count, read.below, read.above) {
                        -2 => f32::NAN,
                        -1 => 0.0,
                        position => values[read.candidate_positions.get(position as usize) as usize][lane],
                    };
                }
            }
            for (gate, &row) in layer.gate_rows.iter().enumerate() {
                let gate_block = projection[layer.output_size + gate];
                let block = &mut projection[row as usize];
                for lane in 0..LANES {
                    block[lane] *= relu32(gate_block[lane]);
                }
            }
            for &row in &layer.boolean_rows {
                for value in projection[row as usize].iter_mut() {
                    *value = evaluate_boolean(*value);
                }
            }
            values[available..available + layer.output_size].copy_from_slice(&projection[..layer.output_size]);
            available += layer.output_size;
        }
    }

    /// [`BatchExecutor::run`] split across `threads` threads (all available cores
    /// when `None`), with the outputs in record order.
    pub fn run_parallel(program: &TensorProgram, sources: &[f32], threads: Option<usize>) -> Vec<f32> {
        let source_size = program.source_size.max(1);
        let records = sources.len() / source_size;
        let threads = threads
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |count| count.get()))
            .clamp(1, records.div_ceil(LANES).max(1));
        // Whole batches per thread, so only the last batch can be partial.
        let records_per_thread = records.div_ceil(threads).div_ceil(LANES) * LANES;
        if threads == 1 || records_per_thread == 0 {
            return BatchExecutor::new(program).run(sources);
        }
        let chunks: Vec<Vec<f32>> = std::thread::scope(|scope| {
            let handles: Vec<_> = sources
                .chunks(records_per_thread * source_size)
                .map(|chunk| scope.spawn(move || BatchExecutor::new(program).run(chunk)))
                .collect();
            handles.into_iter().map(|handle| handle.join().expect("a batch thread panicked")).collect()
        });
        chunks.concat()
    }
}
