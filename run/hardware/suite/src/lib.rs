//! Checks `handwired-rs-run` on a device: a suite file (written on the host by
//! `cargo run -p handwired-rs-run --example hardware_suite`) holds programs, input records and the outputs the host's
//! `TensorExecutor` computed for them, as bits. [`run_suite`] runs every record here and compares every output bit.
//!
//! Suite format, little-endian: magic `HWSUITE\0`, u32 version 1, u32 case count, then per case: u32 name length,
//! the name, u32 program length, the program, u32 record count, then the inputs (f32, records × input cells) and the
//! expected outputs (u32 bits, records × output cells). Names and programs are padded to 4 bytes.

#![no_std]
#![forbid(unsafe_code)]

use handwired_rs_run::{Error, Program};

/// The first 8 bytes of a suite.
pub const MAGIC: [u8; 8] = *b"HWSUITE\0";

/// What one case did.
#[derive(Clone, Copy, Debug)]
pub struct CaseResult<'a> {
    /// The case's name.
    pub name: &'a str,
    /// Why the program or a run failed outright, if it did.
    pub error: Option<Error>,
    /// Records run.
    pub records: usize,
    /// Output cells whose bits differ from the host's.
    pub mismatches: usize,
    /// The first difference: record, output cell, bits here, bits on the host.
    pub first_mismatch: Option<(usize, usize, u32, u32)>,
    /// Cycles per run from the counter given to [`run_suite`]: least, most and total.
    pub cycles: (u32, u32, u64),
    /// The program's size in bytes, layers, values and scratch cells.
    pub program: (usize, usize, usize, usize),
}

/// Why a suite can't be read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SuiteError {
    /// Not a suite, or a version this checker doesn't read.
    NotASuite,
    /// The bytes end early.
    Truncated,
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, length: usize) -> Result<&'a [u8], SuiteError> {
        let end = self.at.checked_add(length).ok_or(SuiteError::Truncated)?;
        let taken = self.bytes.get(self.at..end).ok_or(SuiteError::Truncated)?;
        self.at = end;
        Ok(taken)
    }

    fn u32(&mut self) -> Result<u32, SuiteError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn padded(&mut self, length: usize) -> Result<&'a [u8], SuiteError> {
        let taken = self.take(length)?;
        self.take((4 - length % 4) % 4)?;
        Ok(taken)
    }
}

fn word(bytes: &[u8], index: usize) -> u32 {
    let at = 4 * index;
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

/// Run every case of `suite`, calling `report` with each case's result. `scratch` and `outputs` must be large enough
/// for every program (a case whose program needs more is reported with [`Error::ScratchTooSmall`]); `cycles` reads a
/// free-running cycle counter, e.g. the DWT's, or returns 0 where there is none.
pub fn run_suite<'a>(
    suite: &'a [u8],
    scratch: &mut [f32],
    outputs: &mut [f32],
    mut cycles: impl FnMut() -> u32,
    mut report: impl FnMut(&CaseResult<'a>),
) -> Result<(), SuiteError> {
    let mut reader = Reader { bytes: suite, at: 0 };
    if reader.take(8)? != MAGIC || reader.u32()? != 1 {
        return Err(SuiteError::NotASuite);
    }
    let cases = reader.u32()?;
    for _ in 0..cases {
        let name_length = reader.u32()? as usize;
        let name = core::str::from_utf8(reader.padded(name_length)?).unwrap_or("?");
        let program_length = reader.u32()? as usize;
        let program_bytes = reader.padded(program_length)?;
        let records = reader.u32()? as usize;
        let mut result = CaseResult {
            name,
            error: None,
            records,
            mismatches: 0,
            first_mismatch: None,
            cycles: (u32::MAX, 0, 0),
            program: (program_length, 0, 0, 0),
        };
        let program = match Program::parse(program_bytes) {
            Ok(program) => program,
            Err(error) => {
                // Without the program, the records' size is unknown: the rest of the suite can't be read.
                result.error = Some(error);
                report(&result);
                return Err(SuiteError::Truncated);
            }
        };
        result.program = (program_length, program.layer_count(), program.value_count(), program.scratch_len());
        let (inputs_per, outputs_per) = (program.input_count(), program.output_count());
        let inputs = reader.take(4 * records * inputs_per)?;
        let expected = reader.take(4 * records * outputs_per)?;
        if scratch.len() < program.scratch_len() || outputs.len() < outputs_per {
            result.error = Some(Error::ScratchTooSmall { needed: program.scratch_len(), actual: scratch.len() });
            report(&result);
            continue;
        }
        let mut record_inputs = [0f32; 64];
        if inputs_per > record_inputs.len() {
            result.error = Some(Error::InputLength { expected: record_inputs.len(), actual: inputs_per });
            report(&result);
            continue;
        }
        for record in 0..records {
            for cell in 0..inputs_per {
                record_inputs[cell] = f32::from_bits(word(inputs, record * inputs_per + cell));
            }
            let start = cycles();
            let ran = program.run(
                &record_inputs[..inputs_per],
                &mut scratch[..program.scratch_len()],
                &mut outputs[..outputs_per],
            );
            let spent = cycles().wrapping_sub(start);
            if let Err(error) = ran {
                result.error = Some(error);
                break;
            }
            result.cycles = (result.cycles.0.min(spent), result.cycles.1.max(spent), result.cycles.2 + spent as u64);
            for cell in 0..outputs_per {
                let (got, want) = (outputs[cell].to_bits(), word(expected, record * outputs_per + cell));
                if got != want {
                    result.mismatches += 1;
                    result.first_mismatch.get_or_insert((record, cell, got, want));
                }
            }
        }
        report(&result);
    }
    Ok(())
}
