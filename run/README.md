# handwired-rs-run

Runs circuits exported by [handwired-rs](https://crates.io/crates/handwired-rs) without it: `no_std`, no
allocation, no dependencies, no `unsafe`. It builds for microcontrollers with or without an FPU (tested by building
for Cortex-M0, Cortex-M4F/M7 and RV32IMC) as well as for Linux and other hosts.

You write business logic as an ordinary Rust function with `#[handwired]`, check it against itself with `verify()`,
and export it as bytes. A device runs those bytes with this crate:

- **The same work every time.** Every run does the same operations in the same order, whatever the inputs. On a
  hardware FPU its time is fixed by the program; with software floating point it varies by how long each operation
  takes (up to about 17% on a Cortex-M4, see below).
- **The same results everywhere.** Results are bit-exact with handwired-rs's CPU executor, which handwired-rs checks
  against the original function. That holds wherever f32 arithmetic is IEEE 754, including software floating point.
- **Logic as data.** A program is a file. Its SHA-256 identifies it, so a device can report which logic made a
  decision, and anyone can replay its inputs elsewhere and get the same outputs. Programs can be updated without
  reflashing code.
- **Refuses what it can't do exactly.** Inputs outside the ranges the function declared, or between the steps of
  their grids (200.5 for an `i16`, 0.5 for a `bool` or an enum's variant), are refused, since the circuit is only
  exact on them. Damaged bytes are refused at load time (checksum and structure), so a run never
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
- Checked on hardware:
  - An STM32WL55 (Cortex-M4 without an FPU, so software floating point) ran 8 programs on 62 records each.
  - Every output bit matched the host.
  - The programs cover integer logic, bitwise operations, division, enums, fractional floats, subnormal numbers,
    gathers and a 64-bit hash.
  - QEMU's Cortex-M0 and Cortex-M3 give the same results, and CI runs them; see `hardware/` to repeat the check.
- Run time with software floating point varies with the values, even though the operations don't: each soft-float
  add or multiply takes longer or shorter depending on its operands. On the WL55 the spread between the fastest and
  slowest record was 0.4% for the heater logic and up to 17% for bit-heavy programs. A worst case still exists
  (the operation count times the slowest soft-float operation), but measure it per program.
- With a hardware FPU, run time is nearly fixed. On an STM32H563 (Cortex-M33 with an FPU) 12 benchmark programs
  ran 102 records each. Taking the fastest of 5 timed runs per record, so that stray stalls such as the debugger
  reading over the bus drop out, each program's slowest record took 3 to 24 cycles longer than its fastest, out of
  27,900 to 2.57 million (at most 0.03%). The same functions compiled natively varied by up to 54%, and run through
  the wasmi Wasm interpreter by up to 35%. The runner avoids branching on values to get there: ReLU, table reads and
  the input range check use masks and f32 comparisons, so nothing waits on a branch whose direction depends on the
  data, or on f64 arithmetic that an FPU without f64 does in software.

## Cost

A run takes about 1.3x as long as handwired-rs's in-memory executor on the same machine, because it reads the
program from its bytes as it goes. Measured times:

| Program | Size | Scratch (f32 cells) | Desktop core | STM32WL55, 32 MHz, software floats |
|---|---|---|---|---|
| Heater logic (26 layers) | 11.8 KB | 556 | | 54k cycles, 1.7 ms |
| Unit plan with enums and Options | 3.8 KB | 140 | | 65k cycles, 2.0 ms |
| Division (tables and long division) | 20.9 KB | 573 | | 203k cycles, 6.3 ms |
| 1,129-operation 64-bit hash | 84 KB | 1,190 | about 21 µs | 1.87M cycles, 58 ms |

A circuit evaluates every path, so the same logic in plain Rust is much faster. On the H563, the 12 benchmark
programs took 136 to 10,600 times the cycles of native code, and 4 to 290 times those of wasmi. What this buys is
a fixed list of operations, exactness and the program as data, not speed. The runner itself is about 6 KB of
code.

## Format

Version 2, little-endian; see `handwired_rs::export` (`src/export.rs`) for the layout. Version 1 programs, which
carry no grids, still run, checked against their ranges only. This crate refuses other versions.
