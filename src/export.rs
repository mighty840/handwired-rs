//! A compiled program as bytes, for the `handwired-rs-run` crate to run without this one: on a microcontroller, in a
//! Linux image, anywhere with f32 arithmetic.
//!
//! The format, version 1, is little-endian throughout:
//!
//! ```text
//! magic        8 bytes, "HWCIRC\0\0"
//! version      u32, 1
//! description  u32 length, then that many bytes of UTF-8
//! header       u32 source_size, value_count, gather_slot_count, root_count, layer_count
//! sources      source_size × (f64 min, f64 max), each source cell's declared range
//! roots        root_count × u32, the packed position of each output
//! layers       layer_count × layer
//! checksum     u32, FNV-1a over every byte before it
//!
//! layer        u32 output_size, rows, entries, gates, booleans, computes, gather_reads, index_reads
//!              rows × f32          biases
//!              (rows + 1) × u32    row offsets into the entries
//!              entries × u32       packed positions each entry reads
//!              entries × f32       weights
//!              gates × u32         rows multiplied by relu of row output_size + k
//!              booleans × u32      rows rounded to 0 or 1
//!              computes × (u32 slot, u32 count, count × u32 bit positions)
//!              gather_reads × (u32 row, u32 slot, u32 masked, u32 count, count × u32 candidate positions)
//!              index_reads × (u32 row, u32 index_row, u32 flags, u32 count, count × u32 candidate positions),
//!              flags bit 0: zero below the table, bit 1: zero above it
//! ```
//!
//! Packed positions `0..source_size` are the source cells; each layer appends its `output_size` values after the
//! ones before it. The checksum catches corrupted storage, not tampering: a program's identity is a hash of its
//! file, e.g. its SHA-256.

use crate::tensor::TensorProgram;

/// The first 8 bytes of every exported program.
pub const PROGRAM_MAGIC: [u8; 8] = *b"HWCIRC\0\0";
/// The format version this crate writes.
pub const PROGRAM_FORMAT_VERSION: u32 = 1;

/// FNV-1a over `bytes`, the format's checksum.
pub fn program_checksum(bytes: &[u8]) -> u32 {
    bytes.iter().fold(0x811C_9DC5u32, |hash, &byte| (hash ^ byte as u32).wrapping_mul(0x0100_0193))
}

struct Writer(Vec<u8>);

impl Writer {
    fn u32(&mut self, value: u32) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }
    fn count(&mut self, value: usize) {
        self.u32(u32::try_from(value).expect("a program part has more than 2^32 entries"));
    }
    fn u32s(&mut self, values: &[u32]) {
        values.iter().for_each(|&value| self.u32(value));
    }
    fn f32s(&mut self, values: &[f32]) {
        values.iter().for_each(|value| self.0.extend_from_slice(&value.to_le_bytes()));
    }
    fn f64(&mut self, value: f64) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }
}

impl TensorProgram {
    /// The program in the format above, with `description` (e.g. the function's name and signature) inside, for
    /// `handwired-rs-run`. Its sources' declared ranges go with it, and the runner refuses inputs outside them.
    pub fn to_bytes(&self, description: &str) -> Vec<u8> {
        let mut out = Writer(Vec::new());
        out.0.extend_from_slice(&PROGRAM_MAGIC);
        out.u32(PROGRAM_FORMAT_VERSION);
        out.count(description.len());
        out.0.extend_from_slice(description.as_bytes());
        for count in [self.source_size, self.value_count, self.gather_slot_count, self.roots.len(), self.layers.len()] {
            out.count(count);
        }
        for range in &self.ranges[..self.source_size] {
            out.f64(range.min);
            out.f64(range.max);
        }
        for root in &self.roots {
            let position = self.wire_order.iter().position(|wire| wire == root).expect("a root was removed");
            out.count(position);
        }
        for layer in &self.layers {
            let entries = layer.columns.len();
            for count in [
                layer.output_size,
                layer.bias.len(),
                entries,
                layer.gate_rows.len(),
                layer.boolean_rows.len(),
                layer.gather_computes.len(),
                layer.gather_reads.len(),
                layer.index_reads.len(),
            ] {
                out.count(count);
            }
            out.f32s(&layer.bias);
            out.u32s(&layer.row_offsets);
            let positions: Vec<u32> =
                layer.columns.iter().map(|&column| layer.input_indices[column as usize]).collect();
            out.u32s(&positions);
            out.f32s(&layer.weights);
            out.u32s(&layer.gate_rows);
            out.u32s(&layer.boolean_rows);
            for compute in &layer.gather_computes {
                out.u32(compute.slot);
                out.count(compute.bit_positions.len());
                out.u32s(&compute.bit_positions);
            }
            for read in &layer.gather_reads {
                out.u32s(&[read.row, read.slot, read.masked as u32]);
                out.count(read.candidate_positions.len());
                out.u32s(&read.candidate_positions);
            }
            for read in &layer.index_reads {
                out.u32s(&[read.row, read.index_row, read.below_zero as u32 | (read.above_zero as u32) << 1]);
                out.count(read.candidate_positions.len());
                out.u32s(&read.candidate_positions);
            }
        }
        let checksum = program_checksum(&out.0);
        out.u32(checksum);
        out.0
    }
}
