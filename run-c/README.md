# handwired-rs-run-c

A C interface to [handwired-rs-run](../run), for PLC runtimes and firmware written in C or C++. It runs program files
exported by handwired-rs with the same results and the same refusals as the Rust runner, and its error texts are the
words `handwired-run` prints.

- **No allocation.** The caller owns every buffer: the program's bytes, its storage (`hw_program`), scratch memory,
  inputs and outputs. Parse once at start-up; run every scan.
- **The runner stays as it is.** handwired-rs-run has no `unsafe` code. Turning C pointers into slices happens here,
  in one small crate.
- **Hosted or bare metal.** The default build links Rust's standard library, for a runtime on Linux. For firmware,
  build it `no_std` with `--no-default-features --features panic-halt`. CI builds it for Cortex-M0, Cortex-M4F and
  RISC-V.

## Use

```sh
cargo build --release -p handwired-rs-run-c    # target/release/libhandwired_rs_run_c.a
cc -Irun-c/include app.c target/release/libhandwired_rs_run_c.a -lm -lpthread -ldl
```

```c
#include "handwired_run.h"

hw_program program;
hw_error error;
char text[160];
if (hw_program_parse(bytes, length, &program, &error) != HW_OK) {  /* checksum, format, every index */
  hw_error_text(&error, text, sizeof text);
}
float inputs[3] = {180, 21, 1}, outputs[2];
float scratch[600];                                                  /* hw_program_scratch(&program) cells */
if (hw_program_run(&program, inputs, 3, scratch, 600, outputs, 2, &error) != HW_OK) {
  hw_error_text(&error, text, sizeof text);  /* e.g. "input cell 0 is 215.5, not a multiple of 1, the step ..." */
}
```

`bytes` must stay in place, unchanged, while `program` is used. A refused run doesn't write `outputs`.
`include/handwired_run.h` lists every function and code.

## Tests

`tests/ffi.rs` checks the interface against the Rust runner on every accepted input of the heater program, and
checks its refusals, wrong buffer lengths, damaged bytes and text cut to fit. `tests/smoke.c` does the same from C.
CI compiles it with `-std=c99 -pedantic -Werror` and runs it. The 4diac function block in [`plc/4diac`](../plc/4diac)
is built on this interface.
