//! A C interface to [handwired-rs-run](https://crates.io/crates/handwired-rs-run), for PLC runtimes and firmware
//! written in C or C++. `include/handwired_run.h` declares it. The caller owns every buffer: a program's storage,
//! its scratch memory, its inputs and outputs, and the bytes of the program itself, which must outlive it. Nothing
//! allocates, so a runtime can parse a program once at start-up and run it every scan.
//!
//! ```c
//! hw_program program;
//! hw_error error;
//! if (hw_program_parse(bytes, length, &program, &error) != HW_OK) {
//!     char text[160];
//!     hw_error_text(&error, text, sizeof text);
//! }
//! float inputs[3] = {180, 21, 0}, outputs[2];
//! float scratch[600]; /* at least hw_program_scratch(&program) cells */
//! int code = hw_program_run(&program, inputs, 3, scratch, 600, outputs, 2, &error);
//! ```
//!
//! The runner itself has no `unsafe` code; this crate is the one place where raw pointers from C become slices.

#![no_std]
#![deny(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

use core::fmt::{self, Write};
use core::mem::{align_of, size_of};
use core::{ptr, slice};

use handwired_rs_run::{Error, Program};

/// Storage for a parsed program: `hw_program` in the header. Valid only after [`hw_program_parse`] returns
/// `HW_OK`, and only while the program's bytes stay where they were.
#[repr(C)]
pub struct HwProgram {
    opaque: [u64; 24],
}

/// Why a call failed: `hw_error` in the header. [`hw_error_text`] describes it.
#[repr(C)]
pub struct HwError {
    opaque: [u64; 8],
}

/// What an [`HwProgram`] holds: a marker that [`hw_program_parse`] filled it, then the program.
#[repr(C)]
struct Stored {
    marker: u64,
    program: Program<'static>,
}

/// Written by [`hw_program_parse`], so a run can refuse storage that no parse filled.
const PARSED: u64 = 0x6877_7275_6e5f_6f6b;

/// What an [`HwError`] holds.
#[derive(Clone, Copy)]
enum Failure {
    None,
    NullArgument,
    NotParsed,
    Run(Error),
}

const _: () = assert!(size_of::<Stored>() <= size_of::<HwProgram>() && align_of::<Stored>() <= align_of::<HwProgram>());
const _: () = assert!(size_of::<Failure>() <= size_of::<HwError>() && align_of::<Failure>() <= align_of::<HwError>());

/// The codes calls return, as in the header.
const HW_OK: i32 = 0;
const HW_NULL_ARGUMENT: i32 = 1;
const HW_NOT_PARSED: i32 = 2;

fn code(failure: &Failure) -> i32 {
    match failure {
        Failure::None => HW_OK,
        Failure::NullArgument => HW_NULL_ARGUMENT,
        Failure::NotParsed => HW_NOT_PARSED,
        Failure::Run(error) => match error {
            Error::Truncated => 3,
            Error::NotAProgram => 4,
            Error::UnsupportedVersion(_) => 5,
            Error::Checksum => 6,
            Error::Malformed(_) => 7,
            Error::InputLength { .. } => 8,
            Error::InputOutOfRange { .. } => 9,
            Error::InputOffGrid { .. } => 10,
            Error::InputOutsideVariant { .. } => 11,
            Error::InputNotAllowed { .. } => 12,
            Error::ScratchTooSmall { .. } => 13,
            Error::OutputLength { .. } => 14,
            _ => 15,
        },
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Failure::None => write!(f, "no error"),
            Failure::NullArgument => write!(f, "a null pointer where a buffer was needed"),
            Failure::NotParsed => write!(f, "the program's storage wasn't filled by hw_program_parse"),
            Failure::Run(error) => write!(f, "{error}"),
        }
    }
}

/// Record `failure` in `error`, if the caller gave one, and return its code.
///
/// # Safety
///
/// `error` is null or points to writable storage for an [`HwError`].
unsafe fn report(error: *mut HwError, failure: Failure) -> i32 {
    if !error.is_null() {
        ptr::write(error.cast::<Failure>(), failure);
    }
    code(&failure)
}

/// The slice at `data`, allowing null when `len` is 0.
///
/// # Safety
///
/// Unless `len` is 0, `data` points to `len` readable values.
unsafe fn slice_at<'a, T>(data: *const T, len: usize) -> Option<&'a [T]> {
    match (data.is_null(), len) {
        (_, 0) => Some(&[]),
        (true, _) => None,
        (false, _) => Some(slice::from_raw_parts(data, len)),
    }
}

/// The mutable slice at `data`, allowing null when `len` is 0.
///
/// # Safety
///
/// Unless `len` is 0, `data` points to `len` writable values that nothing else reads or writes during the call.
unsafe fn slice_at_mut<'a, T>(data: *mut T, len: usize) -> Option<&'a mut [T]> {
    match (data.is_null(), len) {
        (_, 0) => Some(&mut []),
        (true, _) => None,
        (false, _) => Some(slice::from_raw_parts_mut(data, len)),
    }
}

/// The program in `program`, if a parse filled it.
///
/// # Safety
///
/// `program` is null or points to an [`HwProgram`] that [`hw_program_parse`] filled, or that holds anything but
/// its marker.
unsafe fn parsed<'a>(program: *const HwProgram) -> Result<&'a Program<'static>, Failure> {
    if program.is_null() {
        return Err(Failure::NullArgument);
    }
    let stored = &*program.cast::<Stored>();
    if stored.marker != PARSED {
        return Err(Failure::NotParsed);
    }
    Ok(&stored.program)
}

/// Check a program's `length` bytes at `bytes` and fill `program` with it. Returns `HW_OK`, or the code of what's
/// wrong with the bytes, which is also written to `error` unless that's null. On failure `program` can't be run.
///
/// # Safety
///
/// `bytes` points to `length` readable bytes that stay unchanged, at the same address, for as long as `program` is
/// used; `program` points to writable storage for an [`HwProgram`]; `error` is null or points to writable storage
/// for an [`HwError`].
#[no_mangle]
pub unsafe extern "C" fn hw_program_parse(
    bytes: *const u8,
    length: usize,
    program: *mut HwProgram,
    error: *mut HwError,
) -> i32 {
    if program.is_null() {
        return report(error, Failure::NullArgument);
    }
    let stored = program.cast::<Stored>();
    // Whatever happens, the storage stops counting as parsed.
    ptr::addr_of_mut!((*stored).marker).write(0);
    let Some(bytes) = slice_at(bytes, length) else {
        return report(error, Failure::NullArgument);
    };
    match Program::parse(bytes) {
        Ok(parsed) => {
            stored.write(Stored { marker: PARSED, program: parsed });
            report(error, Failure::None)
        }
        Err(failure) => report(error, Failure::Run(failure)),
    }
}

/// The program's input cells, or 0 if `program` wasn't filled by a parse.
///
/// # Safety
///
/// As for [`hw_program_run`]'s `program`.
#[no_mangle]
pub unsafe extern "C" fn hw_program_inputs(program: *const HwProgram) -> usize {
    parsed(program).map_or(0, |program| program.input_count())
}

/// The program's output cells, or 0 if `program` wasn't filled by a parse.
///
/// # Safety
///
/// As for [`hw_program_run`]'s `program`.
#[no_mangle]
pub unsafe extern "C" fn hw_program_outputs(program: *const HwProgram) -> usize {
    parsed(program).map_or(0, |program| program.output_count())
}

/// The f32 cells of scratch memory a run needs, or 0 if `program` wasn't filled by a parse.
///
/// # Safety
///
/// As for [`hw_program_run`]'s `program`.
#[no_mangle]
pub unsafe extern "C" fn hw_program_scratch(program: *const HwProgram) -> usize {
    parsed(program).map_or(0, |program| program.scratch_len())
}

/// Write the program's description (its function's name and signature) to `buffer`, as for [`hw_error_text`].
///
/// # Safety
///
/// As for [`hw_program_run`]'s `program`, and [`hw_error_text`]'s `buffer`.
#[no_mangle]
pub unsafe extern "C" fn hw_program_description(program: *const HwProgram, buffer: *mut u8, size: usize) -> usize {
    match parsed(program) {
        Ok(program) => write_text(buffer, size, format_args!("{}", program.description())),
        Err(failure) => write_text(buffer, size, format_args!("{failure}")),
    }
}

/// Run the program on `input_count` input cells, writing its `output_count` output cells. `scratch` holds at
/// least [`hw_program_scratch`] cells. Returns `HW_OK`, or the code of the refusal, which is also written to
/// `error` unless that's null: an input outside its declared range, off its grid or not an encoding of its type,
/// or a buffer of the wrong length. A refused run doesn't write `outputs`.
///
/// # Safety
///
/// `program` is null or points to an [`HwProgram`] that [`hw_program_parse`] filled, whose bytes are still in
/// place; `inputs` points to `input_count` readable cells, `scratch` to `scratch_count` and `outputs` to
/// `output_count` writable cells, none of them overlapping; `error` is null or points to writable storage for an
/// [`HwError`]. A pointer may be null where its count is 0.
#[no_mangle]
pub unsafe extern "C" fn hw_program_run(
    program: *const HwProgram,
    inputs: *const f32,
    input_count: usize,
    scratch: *mut f32,
    scratch_count: usize,
    outputs: *mut f32,
    output_count: usize,
    error: *mut HwError,
) -> i32 {
    let program = match parsed(program) {
        Ok(program) => program,
        Err(failure) => return report(error, failure),
    };
    let (Some(inputs), Some(scratch), Some(outputs)) =
        (slice_at(inputs, input_count), slice_at_mut(scratch, scratch_count), slice_at_mut(outputs, output_count))
    else {
        return report(error, Failure::NullArgument);
    };
    // The runner checks every length and input before it computes anything, and writes `outputs` last.
    match program.run(inputs, scratch, outputs) {
        Ok(()) => report(error, Failure::None),
        Err(failure) => report(error, Failure::Run(failure)),
    }
}

/// Write a description of `error` to `buffer`, the same words `handwired-run` prints, cut to fit `size` bytes with
/// a terminating NUL. Returns the length of the whole description, so a return of `size` or more means it was cut.
///
/// # Safety
///
/// `error` is null or points to an [`HwError`] that a call of this crate wrote; `buffer` is null or points to
/// `size` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn hw_error_text(error: *const HwError, buffer: *mut u8, size: usize) -> usize {
    let failure = if error.is_null() { Failure::NullArgument } else { *error.cast::<Failure>() };
    write_text(buffer, size, format_args!("{failure}"))
}

/// Writes text into a C buffer, keeping the last byte for the NUL and counting what didn't fit.
struct Text {
    buffer: *mut u8,
    size: usize,
    length: usize,
}

impl Write for Text {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        for &byte in text.as_bytes() {
            if self.length + 1 < self.size {
                // SAFETY: `length + 1 < size`, within the buffer the caller gave.
                unsafe { self.buffer.add(self.length).write(byte) };
            }
            self.length += 1;
        }
        Ok(())
    }
}

/// Write `arguments` to `size` bytes at `buffer` with a terminating NUL; returns the full length.
///
/// # Safety
///
/// `buffer` is null or points to `size` writable bytes.
unsafe fn write_text(buffer: *mut u8, size: usize, arguments: fmt::Arguments<'_>) -> usize {
    let size = if buffer.is_null() { 0 } else { size };
    let mut text = Text { buffer, size, length: 0 };
    let _ = text.write_fmt(arguments);
    if size > 0 {
        buffer.add(text.length.min(size - 1)).write(0);
    }
    text.length
}

#[cfg(all(feature = "panic-halt", not(feature = "std"), not(test)))]
#[panic_handler]
fn halt(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
