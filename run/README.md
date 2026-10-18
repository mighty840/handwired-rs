# handwired-rs-run

Runs circuits exported by [handwired-rs](https://crates.io/crates/handwired-rs) without it: `no_std`, no
allocation, no dependencies, no `unsafe`. It builds for microcontrollers with or without an FPU (tested by building
for Cortex-M0, Cortex-M4F/M7 and RV32IMC) as well as for Linux and other hosts.

You write business logic as an ordinary Rust function with `#[handwired]`, check it against itself with `verify()`,
and export it as bytes. A device runs those bytes with this crate:

- **The same work every time.** Every run does the same operations in the same order, whatever the inputs, so how
  long it takes depends only on the program (on arithmetic whose speed doesn't depend on the values; see below).
- **The same results everywhere.** Results are bit-exact with handwired-rs's CPU executor, which handwired-rs checks
  against the original function. That holds wherever f32 arithmetic is IEEE 754, including software floating point.
- **Logic as data.** A program is a file. Its SHA-256 identifies it, so a device can report which logic made a
  decision, and anyone can replay its inputs elsewhere and get the same outputs. Programs can be updated without
  reflashing code.
- **Refuses what it can't do exactly.** Inputs outside the ranges the function declared are refused, since the
  circuit is only exact within them. Damaged bytes are refused at load time (checksum and structure), so a run never
  reads out of bounds.

## Export on the host

```rust
use handwired_rs::handwired;

/// A heater's duty from a temperature in tenths of a degree and a setpoint in degrees.
#[handwired]
fn heater(#[range(-400, 1250)] tenths: i16, #[range(5, 30)] setpoint: i16, boost: bool) -> (u8, bool) {
    let error = setpoint * 10 - tenths;
    let duty = if error <= 0 { 0 } else if error > 100 { 100 } else { error as u8 };
    (if boost { duty.saturating_add(20).min(100) } else { duty }, error < -50)
}

fn main() {
    let compiled = heater_circuit();
    compiled.verify(100_000).unwrap();
    std::fs::write("heater.hwc", compiled.export("heater")).unwrap();
}
```

## Run on the device

```rust,ignore
#![no_std]
use handwired_rs_run::Program;

static HEATER: &[u8] = include_bytes!("heater.hwc");

fn step(tenths: i16, setpoint: i16, boost: bool) -> Result<(u8, bool), handwired_rs_run::Error> {
    let program = Program::parse(HEATER)?; // once, at startup, in real firmware
    let mut scratch = [0.0f32; 512]; // at least program.scratch_len()
    let mut outputs = [0.0f32; 2];
    let inputs = [tenths as f32, setpoint as f32, boost as u8 as f32];
    program.run(&inputs, &mut scratch[..program.scratch_len()], &mut outputs)?;
    Ok((outputs[0] as u8, outputs[1] != 0.0))
}
```

Inputs and outputs are f32 cells, in the order handwired-rs's `HwType::encode` gives them. Integers and bools are one
cell each; arrays, tuples and structs are their parts in order; an enum is its variant's index, then every variant's
fields. `Program::description()` holds the function's name and its parameter and result types.

## On Linux: `handwired-run`

With the `std` feature, `handwired-run heater.hwc` reads one record of input cells per line from standard input
(separated by spaces or commas) and prints the output cells, and `handwired-run --info heater.hwc` describes a
program:

```text
$ printf '200 21 0\n215 20 1\n' | handwired-run heater.hwc
10 0
20 0
```

## Exactness and timing on hardware

- Results match the CPU executor bit for bit where f32 add and multiply are IEEE 754 with round-to-nearest. That is
  true of hardware FPUs in their default mode and of Rust's software floating point. Programs that keep fractional
  values can reach subnormal numbers. A flush-to-zero mode, which some startup code enables, then changes results;
  integer logic never makes subnormals.
- Run time is fixed by the program on hardware FPUs. Software floating point and some FPUs take longer on subnormal
  numbers, so time can vary there by a few cycles per operation.
- On microcontrollers this is checked by building only: the tests that compare results bit for bit run on the host.
  Software floating point is IEEE-exact, so the host results carry over, but nothing has run on real hardware yet.

## Cost

A run is about 1.3x the time of handwired-rs's in-memory executor on the same machine, since it reads the program
from its bytes as it goes. On a desktop core, a 1,129-operation 64-bit hash takes about 21 µs. Its program is 84 KB,
and the scratch it needs is 1,190 f32 cells. A circuit evaluates every path, so the same logic in plain Rust is
much faster: what this buys is a fixed operation count, exactness and the program as data, not speed.

## Format

Version 1, little-endian; see `handwired_rs::export` (`src/export.rs`) for the layout. This crate refuses other
versions.
