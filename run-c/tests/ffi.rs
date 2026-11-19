//! The C interface against the runner it wraps: the same outputs, the same refusals and words, and every buffer the
//! caller gets wrong refused, on the heater program the Yocto layer installs.

use handwired_rs_run::Program;
use handwired_rs_run_c::*;
use std::mem::MaybeUninit;

const HEATER: &[u8] = include_bytes!("../../meta-handwired/recipes-handwired/handwired-heater/files/heater.hwc");

fn parse(bytes: &[u8]) -> (i32, Box<HwProgram>, String) {
    let mut program = zeroed_program();
    let mut error = MaybeUninit::<HwError>::uninit();
    let code = unsafe { hw_program_parse(bytes.as_ptr(), bytes.len(), &mut *program, error.as_mut_ptr()) };
    (code, program, text_of(error.as_ptr()))
}

/// Storage no parse has filled. Zeroed rather than uninitialised, so the tests may hold references to it.
fn zeroed_program() -> Box<HwProgram> {
    // SAFETY: an HwProgram is an array of integers, for which all zeros is a value.
    Box::new(unsafe { MaybeUninit::zeroed().assume_init() })
}

fn text_of(error: *const HwError) -> String {
    let mut buffer = [0u8; 200];
    let length = unsafe { hw_error_text(error, buffer.as_mut_ptr(), buffer.len()) };
    assert!(length < buffer.len());
    String::from_utf8(buffer[..length].to_vec()).unwrap()
}

/// Runs the program on `inputs` with exactly-sized buffers; returns the code, the outputs and the error's words.
fn run(program: &HwProgram, inputs: &[f32]) -> (i32, Vec<f32>, String) {
    let mut scratch = vec![0.0f32; unsafe { hw_program_scratch(program) }];
    let mut outputs = vec![-1.0f32; unsafe { hw_program_outputs(program) }];
    let mut error = MaybeUninit::<HwError>::uninit();
    let code = unsafe {
        hw_program_run(
            program,
            inputs.as_ptr(),
            inputs.len(),
            scratch.as_mut_ptr(),
            scratch.len(),
            outputs.as_mut_ptr(),
            outputs.len(),
            error.as_mut_ptr(),
        )
    };
    (code, outputs, text_of(error.as_ptr()))
}

#[test]
fn a_parsed_program_describes_itself() {
    let (code, program, text) = parse(HEATER);
    assert_eq!((code, text.as_str()), (0, "no error"));
    let native = Program::parse(HEATER).unwrap();
    unsafe {
        assert_eq!(hw_program_inputs(&*program), 3);
        assert_eq!(hw_program_outputs(&*program), 2);
        assert_eq!(hw_program_scratch(&*program), native.scratch_len());
        let mut buffer = [0u8; 256];
        let length = hw_program_description(&*program, buffer.as_mut_ptr(), buffer.len());
        assert_eq!(&buffer[..length], native.description().as_bytes());
        assert_eq!(buffer[length], 0);
    }
}

#[test]
fn runs_give_the_runners_outputs() {
    let (_, program, _) = parse(HEATER);
    let native = Program::parse(HEATER).unwrap();
    for (inputs, expected) in [
        ([180.0, 21.0, 0.0], [30.0, 0.0]),
        ([180.0, 21.0, 1.0], [50.0, 0.0]),
        ([50.0, 21.0, 0.0], [100.0, 0.0]),
        ([306.0, 5.0, 1.0], [20.0, 1.0]),
    ] {
        let (code, outputs, _) = run(&program, &inputs);
        assert_eq!(code, 0);
        assert_eq!(outputs, expected);
        assert_eq!(outputs, native.evaluate(&inputs).unwrap());
    }
    // Every accepted input, bit for bit.
    for tenths in -400..=1250 {
        for setpoint in [5, 17, 30] {
            let inputs = [tenths as f32, setpoint as f32, (tenths & 1) as f32];
            let (code, outputs, _) = run(&program, &inputs);
            assert_eq!(code, 0);
            let expected = native.evaluate(&inputs).unwrap();
            assert_eq!(
                outputs.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn refusals_carry_the_runners_codes_and_words_and_leave_outputs_alone() {
    let (_, program, _) = parse(HEATER);
    let native = Program::parse(HEATER).unwrap();
    for (inputs, expected_code) in [([2005.0, 21.0, 0.0], 9), ([215.5, 21.0, 0.0], 10), ([180.0, 21.0, 2.0], 9)] {
        let (code, outputs, text) = run(&program, &inputs);
        assert_eq!(code, expected_code);
        assert_eq!(text, native.evaluate(&inputs).unwrap_err().to_string());
        assert_eq!(outputs, [-1.0, -1.0]);
    }
}

#[test]
fn buffers_of_the_wrong_length_are_refused() {
    let (_, program, _) = parse(HEATER);
    let mut scratch = vec![0.0f32; 4096];
    let mut outputs = [0.0f32; 4];
    let inputs = [180.0f32, 21.0, 0.0];
    let mut error = MaybeUninit::<HwError>::uninit();
    let mut call = |inputs: &[f32], scratch: &mut [f32], outputs: &mut [f32]| unsafe {
        let code = hw_program_run(
            &*program,
            inputs.as_ptr(),
            inputs.len(),
            scratch.as_mut_ptr(),
            scratch.len(),
            outputs.as_mut_ptr(),
            outputs.len(),
            error.as_mut_ptr(),
        );
        (code, text_of(error.as_ptr()))
    };
    assert_eq!(call(&inputs[..2], &mut scratch, &mut outputs[..2]).0, 8);
    assert_eq!(call(&inputs, &mut scratch[..10], &mut outputs[..2]).0, 13);
    assert_eq!(call(&inputs, &mut scratch, &mut outputs).1, "4 output cells for a program of 2");
    // A null buffer with cells to hold, and no error storage at all.
    let code = unsafe {
        hw_program_run(
            &*program,
            std::ptr::null(),
            3,
            scratch.as_mut_ptr(),
            4096,
            outputs.as_mut_ptr(),
            2,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(code, 1);
}

#[test]
fn damaged_or_missing_bytes_are_refused() {
    let mut changed = HEATER.to_vec();
    changed[100] ^= 1;
    assert_eq!(parse(&changed).0, 6);
    assert_eq!(parse(&HEATER[..40]).0, 6);
    assert_eq!(parse(&HEATER[..3]).0, 3);
    assert_eq!(parse(b"not a program at all, but long enough to look").0, 4);
    let mut program = zeroed_program();
    let mut error = MaybeUninit::<HwError>::uninit();
    let code = unsafe { hw_program_parse(std::ptr::null(), 10, &mut *program, error.as_mut_ptr()) };
    assert_eq!((code, text_of(error.as_ptr()).as_str()), (1, "a null pointer where a buffer was needed"));
    // The failed parse left storage that can't be run.
    let (code, _, text) = run(&program, &[180.0, 21.0, 0.0]);
    assert_eq!((code, text.as_str()), (2, "the program's storage wasn't filled by hw_program_parse"));
}

#[test]
fn a_failed_parse_unmarks_storage_a_good_one_filled() {
    let (_, mut program, _) = parse(HEATER);
    let mut changed = HEATER.to_vec();
    changed[100] ^= 1;
    let code = unsafe { hw_program_parse(changed.as_ptr(), changed.len(), &mut *program, std::ptr::null_mut()) };
    assert_eq!(code, 6);
    assert_eq!(unsafe { hw_program_inputs(&*program) }, 0);
    assert_eq!(run(&program, &[180.0, 21.0, 0.0]).0, 2);
}

#[test]
fn text_is_cut_to_fit_with_a_nul() {
    let (_, program, _) = parse(HEATER);
    let (_, _, full) = run(&program, &[2005.0, 21.0, 0.0]);
    let mut error = MaybeUninit::<HwError>::uninit();
    let mut scratch = vec![0.0f32; 4096];
    let mut outputs = [0.0f32; 2];
    unsafe {
        hw_program_run(
            &*program,
            [2005.0f32, 21.0, 0.0].as_ptr(),
            3,
            scratch.as_mut_ptr(),
            4096,
            outputs.as_mut_ptr(),
            2,
            error.as_mut_ptr(),
        );
        let mut buffer = [0xffu8; 8];
        assert_eq!(hw_error_text(error.as_ptr(), buffer.as_mut_ptr(), buffer.len()), full.len());
        assert_eq!(&buffer[..7], &full.as_bytes()[..7]);
        assert_eq!(buffer[7], 0);
        assert_eq!(hw_error_text(error.as_ptr(), std::ptr::null_mut(), 8), full.len());
        assert_eq!(hw_error_text(error.as_ptr(), buffer.as_mut_ptr(), 0), full.len());
    }
}
