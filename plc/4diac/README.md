# HANDWIRED_RUN for Eclipse 4diac

`HANDWIRED_RUN` is a function block for [Eclipse 4diac FORTE](https://github.com/eclipse-4diac/4diac-forte), the
IEC 61499 runtime. It runs a program file exported by handwired-rs, so logic written as an ordinary Rust function
becomes one block in a control application. The logic ships as a data file that you can replace without rebuilding
FORTE.

| Port | Type | |
|---|---|---|
| `INIT` → `INITO` | event | `QI` = TRUE reads and checks the file at `PATH`, and `QO` says whether it loaded. `QI` = FALSE unloads it |
| `REQ` → `CNF` | event | runs the program on the first cells of `IN` and writes the first cells of `OUT` |
| `QI`, `QO` | BOOL | event qualifiers; `QO` is FALSE when no program is loaded or a run was refused |
| `PATH` | STRING | the program file, relative to FORTE's working directory |
| `IN`, `OUT` | ARRAY [0..15] OF REAL | the program's input and output cells, in the order of its function's parameters and result |
| `STATUS` | STRING | `OK`; after INIT, the program's name and size; or why the file or the inputs were refused |

A run does the same operations in the same order whatever the inputs. Inputs outside the ranges, grids or types the
program's function declared are refused before the run. `QO` is then FALSE, `STATUS` gives the runner's reason, and
`OUT` keeps its previous values, so logic downstream must check `QO`. INIT allocates; REQ doesn't.

## Build

FORTE takes modules in its source tree, so `handwired/` goes into FORTE's `modules/` like its own modules do.
`build.sh` does it all: it clones FORTE 3.3.0, adds the module, builds the C interface ([`run-c`](../../run-c)) and
FORTE, then runs the test application:

```sh
plc/4diac/build.sh /tmp/4diac      # JOBS=4 for a faster build; FORTE_TAG for another release
```

By hand:

1. Copy `handwired/` to `modules/handwired` and add `add_subdirectory(handwired)` to `modules/CMakeLists.txt`.
2. Run `cargo build --release -p handwired-rs-run-c` in handwired-rs.
3. Configure FORTE with `-DFORTE_MODULE_HANDWIRED=ON -DHANDWIRED_RS_DIR=<handwired-rs checkout>`.

`HANDWIRED_RUN.fbt` is the block's type for the 4diac IDE. Import it into a project to place the block in an
application.

## Test

`test/handwired_run.fboot` is a 4diac application that FORTE runs at start-up. It loads `heater.hwc`, the example
program of the Yocto layer, and runs it at 18.0 °C with a setpoint of 21 °C and boost on. It then checks that the
duty is 50 and cooling is off. A second instance runs an input off the integer grid (215.5) and checks that it's
refused, with the runner's words in `STATUS`. CI runs it on every push.
