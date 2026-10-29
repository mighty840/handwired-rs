# handwired-rs

[![CI](https://github.com/mighty840/handwired-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/mighty840/handwired-rs/actions/workflows/ci.yml)

A Rust port of [chenglou/handwired](https://github.com/chenglou/handwired) by Cheng Lou, plus a `#[handwired]` attribute
that compiles ordinary Rust functions into handwired circuits, and a WebGPU backend.

A handwired program is a fixed list of f32 operations: weighted sums, ReLU products, a rounding-to-bit atom and table
reads. It runs as layers of sparse matrix multiplications, the shape of a neural net, and stays differentiable, while
computing exactly what the ordinary code computes. The ideas, the instruction set, the exactness rules and the
comparison constructions are handwired's; this crate ports them to Rust and checks itself against handwired's
published numbers.

```sh
cargo add handwired-rs            # the circuit builder, #[handwired] and the CPU executors
cargo add handwired-rs-gpu        # optional: run compiled programs on a GPU through wgpu
cargo add handwired-rs-run        # optional: run exported programs anywhere, no_std, without handwired-rs
```

**Status:** 0.8, experimental, checked by the tests listed below on one machine and in CI. Read
[Limitations](#limitations) before relying on it.

```rust
use handwired_rs::{emit_compare, Circuit, Relation};

// const result = (a > 4) ? 10 : 12
let mut circuit = Circuit::new(1);
let a = circuit.source(0);
circuit.assume_range(a, 0.0, 100.0);
circuit.assume_grid(a, 1.0);
let four = circuit.constant(4.0);
let above_four = emit_compare(&mut circuit, a, Relation::Greater, four);
let result = circuit.affine(above_four * -2.0 + 12.0);

let program = circuit.compile(Some(&[result]));
```

Affine forms are written with operators on `Wire` (`x * 0.5 - 26.0`, `a - b`) and appended by passing them to a builder
method. Builder methods panic on a contract violation (e.g. comparing an operand with no finite declared range), since
that is a bug in the program being built; evaluation returns `Error` for bad input data.

## `#[handwired]`: compile ordinary Rust

```rust
use handwired_rs::handwired;

#[handwired]
fn toggle_step(on: bool, armed: bool, down: bool, up: bool, cancel: bool,
               #[range(-16384, 16384)] #[grid(1.0 / 64.0)] x: f32,
               #[range(0, 16384)] left: i32) -> (bool, bool) {
    let inside = x >= left as f32 && x < left as f32 + 52.0;
    let mut on = on;
    let mut armed = armed;
    if down { armed = inside; }
    if up && armed && inside { on = !on; }
    if up || cancel { armed = false; }
    (on, armed)
}

let compiled = toggle_step_circuit();          // the circuit, compiled to tensor layers
compiled.verify(20_000).unwrap();              // vs. the original function, on sampled inputs
let (on, armed) = compiled.eval(&(false, false, true, false, false, 10.0, 0)).unwrap();
let records = vec![(false, false, true, false, false, 10.0, 0); 10_000];
let results = compiled.run_batch(&records).unwrap();  // many records: vectorized, on every core
```

The attribute keeps the function as it is, as the oracle, and adds `toggle_step_hw` (the body over circuit values,
which other `#[handwired]` functions call) and `toggle_step_circuit()` (builds and compiles it). The body runs once at
build time with every input-dependent value as a typed `Val<T>`; operators on `Val`s append operations, so ordinary
code like `a + b` or `x.min(y)` stays as written. The macro rewrites only what Rust can't overload: comparisons,
`&&`/`||`, indexing, assignment, `if`, `match` and `as`.

**The contract:** the circuit agrees with the function on every input where the function returns without panicking.
Overflow and out-of-bounds indexing panic in the oracle, so their circuit values are unspecified. `verify` skips
inputs where the oracle panics; run it with overflow checks on (e.g. a debug build).

**Supported:** `bool`, `u8`–`u64`, `i8`–`i64`, `usize`, `f32`, arrays, tuples (up to 12), `Option<T>`, `Result<T, E>`, and
`#[derive(HwType)]` structs and enums (with or without data in their variants) as parameters (up to 16), results and
locals;
`+ - * !`, comparisons, `&& || & | ^`; integer `/` and `%` by a constant or an input value, see
[Division](#division); f32 `/` by a power of two;
integer `& | ^ !`, `<<` and `>>` by a constant or an input value (wrapping like Rust) and `count_ones()`, see
[Bitwise operators](#bitwise-operators);
`min max clamp abs wrapping_add/sub saturating_add/sub`; `as` casts between them (f32 → int truncates and saturates
like Rust); `if`/`else` as a statement or value; `match` with literals, ranges, `|`, guards, tuples, bindings and
consts; `for` over ranges or arrays that don't depend on the inputs (they unroll); `while`, `while let` and `loop`
with `#[max_iterations(N)]`, see [Loops with a limit](#loops-with-a-limit); array reads and writes at computed
indices, including nested arrays; calls to other `#[handwired]` functions; `handwired_rs::dsl::running_sums(array)`
for integers (a loop in the function, a Brent-Kung prefix network of about 2 log2(n) layers in the circuit; call it by
that path, since the macro treats a bare lowercase call as another `#[handwired]` function); `return`, `?`, `break` and `continue` anywhere,
including under conditions that depend on the inputs.

**Early exits:** the trace runs every path, so an exit can't leave it. A function that returns early gets a `done` bit
and a result slot: each `return` records its value where no earlier return happened, and the function's value picks
the recorded one. A loop with `break`/`continue` gets a `halted` bit, and assignments inside it apply only where it
still runs. Under native conditions, e.g. `if index == 3 { break; }`, exits leave the unrolled loop for real, and
selects on flags that no input-dependent branch has set fold away, so they cost nothing. A `return` outside input-dependent
branches and flagged loops also leaves for real.

### Loops with a limit

A circuit runs a fixed number of steps, so a `while`, `while let` or `loop` whose end depends on the inputs declares
how many passes it may take:

```rust
#[handwired]
fn gcd(#[range(1, 255)] a: u8, #[range(1, 255)] b: u8) -> u8 {
    let (mut a, mut b) = (a, b);
    #[max_iterations(254)]
    while a != b {
        if a > b {
            a -= b;
        } else {
            b -= a;
        }
    }
    a
}
```

The circuit's copy is `for _ in 0..254 { if !(a != b) { break; } ... }`, which the `break` machinery above handles:
each pass costs the body plus its flags, so gcd is 6,345 operations in 2,540 layers, and a tight limit matters. The
function keeps its loop and counts its passes: one that would start pass N + 1 panics with "this loop would run more
times than its #[max_iterations(N)]", and `verify` reports those inputs as `VerifyError::IterationLimit` instead of
skipping them like other panics. `break`, `continue`, early `return`, nesting with `for` and `while let` on enums
work; labeled loops and `break value` don't. A loop whose condition doesn't depend on the inputs leaves for real once
it's false, so its limit costs nothing there. The tests run gcd on every pair of u8s and digit sums on every u16.

**Declarations:** `#[range(lo, hi)]` is required on `f32` parameters and on integer types wider than 2^24 (`u32`,
`i32`, ...), since a parameter or result is one f32 cell within ±2^24. Plain `+ - *` on them must stay there too (the
build panics, pointing at the operation, if they may not); shifts, bitwise operators and `wrapping_*` may go beyond,
see [Wide integers](#wide-integers).
`#[grid(step)]` declares a power-of-two grid, e.g. `1.0 / 64.0` for pointer coordinates, which makes comparisons
cheap. On a compound parameter, `#[range]` applies to every scalar, while `#[ranges([(2, 20), (1, 200), (2, 6)])]`, or
`#[ranges(CONST)]` with a `const CONST: [(f64, f64); N]`, gives each scalar its own, in order.

**Structs.** `#[derive(Clone, Copy, Debug, PartialEq, handwired_rs::dsl::HwType)]` on a struct with named fields of
scalars, arrays and other derived structs makes it a parameter, local and result, with `#[range]`, `#[ranges]` and
`#[grid]` on each field:

```rust
#[derive(Clone, Copy, Debug, PartialEq, HwType)]
pub struct Pig {
    #[range(0, 100)]
    pub scores: [u8; 2],
    #[range(0, 1)]
    pub turn: u8,
    #[range(0, 30)]
    pub pot: u8,
}

#[handwired]
pub fn pig_step(state: Pig, #[range(1, 6)] roll: u8, hold: bool) -> Pig {
    let mut next = state;
    if hold {
        next.scores[state.turn as usize] += state.pot;
        next.pot = 0;
        next.turn = 1 - state.turn;
    } else if roll == 1 {
        next.pot = 0;
        next.turn = 1 - state.turn;
    } else {
        next.pot += roll;
    }
    next
}
```

Field reads and writes, including through computed indices (`board.units[i].health = ...`), struct literals with
`..base`, and structs as an `if`'s value all work. The derive adds `PigHw`, the struct's circuit form, which `Pig { .. }`
literals in `#[handwired]` code build. Tuples implement both `HwType` and `Params`, so with both traits imported, encode a parameter
tuple as `Params::encode(&args, &mut cells)`.

**Enums.** The same derive on an enum whose variants carry no data makes it a scalar: one cell holding the variant's
discriminant, declared as the range of its discriminants and sampled only as real variants. `match` on it (paths, bare
variants after `use Phase::*`, `|`, guards, `_`), `==`, `!=`, `matches!(x, A | B if guard)`, `if let A = x`, `as`
casts to integers, ordering comparisons (with a derived `PartialOrd`, which orders by discriminant), enum fields,
arrays of enums read and written at computed indices, and enum results all work. Its circuit form is `Val<Phase>`.
Each arm costs one equality comparison, which the builder shares between `match`es on the same value.

```rust
#[derive(Clone, Copy, Debug, PartialEq, HwType)]
pub enum Phase { Rolling, Held, Busted, Won }

#[derive(Clone, Copy, Debug, PartialEq, HwType)]
pub struct Turn {
    pub phase: Phase,
    #[range(0, 100)]
    pub score: u8,
    #[range(0, 30)]
    pub pot: u8,
}

#[handwired]
pub fn turn_step(turn: Turn, #[range(1, 6)] roll: u8, hold: bool) -> Turn {
    let mut next = turn;
    match turn.phase {
        Phase::Rolling if hold => {
            next.score = turn.score + turn.pot;
            next.pot = 0;
            next.phase = if next.score >= 50 { Phase::Won } else { Phase::Held };
        }
        Phase::Rolling => {
            if roll == 1 {
                next.pot = 0;
                next.phase = Phase::Busted;
            } else {
                next.pot = turn.pot + roll;
            }
        }
        Phase::Held | Phase::Busted => next.phase = Phase::Rolling,
        _ => {}
    }
    next
}
```

**Variants with data, and `Option<T>`.** On an enum whose variants carry data, the derive lays out one cell for the
variant's index, then every variant's fields in order, each in its own cells. A value of one variant holds 0 in the
other variants' cells, so their declared ranges take in 0: `Attack(#[range(1, 6)] u8)` declares its cell 0..6, and
the circuit is exact over that whole range. `Option<T>` works the same way, with a `bool` for `Some` and `T`'s cells.
A parameter's `#[range]` applies to the fields' scalars, and `#[ranges]` gives one range per cell, the variant's index
first.

```rust
#[derive(Clone, Copy, Debug, PartialEq, HwType)]
pub enum Action {
    Wait,
    Move { #[range(-1, 1)] dx: i8, #[range(-1, 1)] dy: i8 },
    Attack(#[range(1, 6)] u8),
}

// `Unit` is a derived struct with `x: i8`, `y: i8` and `hp: u8` (tests/macro_variants.rs has the full example).
#[handwired]
pub fn act(unit: Unit, action: Action) -> (Unit, Option<u8>) {
    let mut next = unit;
    let mut dealt: Option<u8> = None;
    match action {
        Action::Wait => {}
        Action::Move { dx, dy } => {
            next.x = (unit.x + dx).clamp(0, 9);
            next.y = (unit.y + dy).clamp(0, 9);
        }
        Action::Attack(damage) => dealt = Some(damage * 2),
    }
    (next, dealt)
}
```

Patterns bind fields by position or name, nest (`Some(Action::Attack(d)) if d > 3`), and take literals, ranges and
`..` at the end; `match`, `if let`, `while let` and `matches!` all use them. Each variant test is one equality
comparison of the index. `==` and `!=` compare the variant and, for variants with fields, the fields; `<`, `<=`, `>`,
`>=`, `min`, `max` and `clamp` order values as Rust derives it, by variant in declaration order and then field by field
(`None` before `Some`, `Ok` before `Err`), one comparison pair per cell. `Option` values have `is_some`, `is_none`, `unwrap_or`, `unwrap` and `expect` (the last
two give an unspecified value for `None`, where Rust panics). The derive adds `ActionHw`, the circuit form.
Constructors name the variant by its enum, `Action::Attack(d)` or `Action::Move { dx, dy: 0 }`, since the macro can't
tell which enum a bare `Attack(d)` after `use Action::*` belongs to (bare unit variants work, as in patterns). `Some(x)`
takes its type from where it goes (an annotated `let`, a result, an assigned variable, a field), else from `x`, so a literal may
need a suffix, `Some(3u8)`, and `let mut x = None` needs an annotation.

`Result<T, E>` works the same way: a bit for whether it's `Err`, then the `Ok` value's cells and the error's. `Ok(..)`
and `Err(..)` take their type from where they go like `Some`, and results have `is_ok`, `is_err`, `ok`, `err`,
`unwrap_or`, `unwrap`, `expect`, `unwrap_err` and `expect_err`. `?` works on a `Result` in a function that returns a
`Result`, and on an `Option` in one that returns an `Option`; it leaves early like `return`, and the error type must be
the function's own (there's no `From` conversion).

### Bitwise operators

Integer `&`, `|`, `^`, `!`, `<<`, `>>` (and their `op=` forms) and `count_ones()` work on values within ±2^24. A value
is split into its bits once, in two's complement: one step of two ReLUs per bit, one layer per bit up to 11 bits and
two above, where each bit is kept as its own sum so f32 adds exactly. `a & b` is then one ReLU per bit, and `a ^ b = a + b - 2 (a & b)` and `a | b = a + b - (a & b)` are sums on top
of it. A u8 XOR is 40 operations in 10 layers. `!x` is a sum. `>>` by a constant keeps the high bits (Rust's
arithmetic shift), `<<` by a constant is a multiplication where the result fits the type and wraps through the bits
where it may not, and a shift by an input value reads the result for every possible amount. A value's range sets its
bit count, so `#[range(0, 511)] board: u16` splits into 9 bits.

```rust
#[handwired]
fn xorshift16(state: u16) -> u16 {
    let mut x = state;
    x ^= x << 7;
    x ^= x >> 9;
    x ^= x << 8;
    x
}
```

The tests run every u8 and i8 pair, every xorshift16 state and every value of 11- and 12-bit operands through the
circuits. A literal shifted by an input value needs a suffix, e.g. `0x0Fu8 << amount`.

### Wide integers

`u32`, `i32`, `u64`, `i64` and `usize` values may go beyond ±2^24 inside a circuit. Where a shift, a bitwise operator,
a rotate, `wrapping_add/sub/mul`, a constant like `0x9E37_79B9` or an operand that is already wide takes a value
there, it is held as its bytes: one wire per 8 bits of its two's-complement pattern. `wrapping_add` carries from byte
to byte, `wrapping_mul` adds columns of byte products (each below 2^16, so exact) and carries, comparisons are the
carry out of `a + !b + 1` (signed ones flip the top bit first), bitwise operators split bytes into bits, and shifts
and rotates move bits, or whole bytes for multiples of 8. Casts truncate or sign-extend like `as`, and wide values go
through `if`s, `match`es and table reads at computed indices.

```rust
#[handwired]
fn fnv1a(bytes: [u8; 8]) -> [u8; 4] {
    let mut hash = 0x811C_9DC5u32;
    for byte in bytes {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    [hash as u8, (hash >> 8) as u8, (hash >> 16) as u8, (hash >> 24) as u8]
}
```

Parameters and results are still one cell each within ±2^24, so a wide value goes in and out as bytes, or once it
fits again: where its high bytes are known zeros, e.g. after `h >> 24` or `h & 0xFFFF` (or, for signed types, known
255s, e.g. `x | !0xFFFF`), the value gets one wire again, so it can be returned, compared, cast to `f32` or added
without a cast. It keeps its bytes for bitwise operators and shifts, and the circuit drops whichever form nothing
reads. Returning a value that may still leave ±2^24 fails the build with a message saying so. Plain `+`, `-` and `*`
of one-wire values keep failing the build where their declared ranges may leave ±2^24, so a loose `#[range]` isn't
silently turned into byte arithmetic, which costs far more; arithmetic on a value that came back from bytes goes back
to bytes instead, e.g. `(h & 0xFFFF) * 0x10001`. A signed value shifted right, e.g. `x >> 24` on an `i32`, gets one
wire too. `/` and `%` by a constant are long division over the bytes, a byte at a time for divisors up to 65,536 and
fewer bits at a time up to 2^23, truncating toward zero like Rust (a power of two divides an unsigned value with a
shift and a mask), and so are one-wire values whose range is too wide for a table; the remainder, e.g. `h % 1000`,
comes back as one wire. `abs` of a wide value compares it with 0 and negates where needed. The tests check u32, i32
and u64 arithmetic, bitwise operators, shifts by inputs up to 63, rotates, casts, narrowing, division and `abs`
against Rust (sampled over the full width, plus every carry boundary and the extremes), FNV-1a, and a 49-bit Connect
Four bitboard; the GPU test runs a wide hash.

### Division

Integer `/` and `%` (and `/=`, `%=`) truncate toward zero like Rust. By a constant, they read a table of results over
the dividend's range, up to 65,536 entries (long division over bytes beyond that, see above). By a value that depends
on the inputs, small ranges read one table of every pair's result, e.g. `pot / players` with `pot` in 0..200 and
`players` in 1..6 reads 1,206 entries in 2 layers. Otherwise it's long division over the dividend's bits: one
comparison step per bit, which takes the divisor off the running remainder where it fits, on magnitudes with the
signs put back after. A u32 in 0..60,000 divided by a value in 1..300 is 125 operations in 65 layers. Steps where the
remainder can't reach the smallest divisor yet cost nothing. The divisor stays within ±2^23 (larger ranges fail the
build), and the dividend may be a wide value of any width. Dividing by zero panics in Rust, so the circuit's result
there is unspecified and `verify` skips those inputs.

See [Limitations](#limitations) for what isn't supported.

## GPU: `handwired-rs-gpu`

The `gpu/` crate runs a compiled program as WebGPU compute shaders (wgpu 25), one invocation per record, so a batch
of records runs in parallel. It reproduces the CPU reference evaluator bit for bit.

The shader code is straight-line, a few statements per operation, and Mesa's compile time grows much faster than the
code (on RADV: 0.1 s for 1,200 statements, 6 s for 4,900, 17–27 s for 8,400 depending on load). So a program of up to
2,000 statements is one shader, and a longer one is split into stages that compile in parallel and run one after
another: 1,000 statements each up to 20,000, then fewer with the square root of the length, down to 250 from 320,000.
Each cut goes where few values cross it; the values later stages read go through a storage buffer, which costs run
time in proportion to how many cross, and an integer within a small range crosses as only the bits its range needs,
packed into 32-bit words with others (roughly 30% faster for the wide hash below, 20% for a bit-heavy XOR, whose ReLU
ramps mostly have ranges too loose to pack). On RADV, that 8,400-statement program, a 1,131-operation wide hash, becomes 6
stages that set up in 0.47 s and run 262,144 records in about 62 ms, against 32 ms as one shader (in stages of 250:
0.09 s and 120 ms). SHA-1 (68,036 operations, about 460,000 statements, [below](#two-larger-examples)) becomes 1,240
stages of about 250 that set up in 8–17 s depending on load, and hashes the 2,000 messages in 0.3–0.5 s, measured
with a GPU copy of the example. Mesa also caches compiled shaders on disk, so setting up the same program
again is faster (2.6 s for SHA-1). `GpuBatch::staged` and `GpuProgram::from_program_staged` take another stage size:
larger stages run faster and compile slower, and `usize::MAX` gives one shader.

```rust
let gpu = handwired_rs_gpu::Gpu::new()?;
let batch = handwired_rs_gpu::GpuBatch::new(&gpu, &find_pattern_circuit())?;
let results: Vec<(bool, u8)> = batch.run(&gpu, &records)?;   // records: Vec<([u8; 32], [u8; 3])>
```

Shader compilers break f32 exactness in ways handwired measured on Metal, and that the tests here reproduce on AMD
(RADV) and Mesa's lavapipe: they fuse `a + w * x` into a multiply-add and regroup sums. The generated WGSL wraps every
inexact product and every addition in `max(x, lowest)`, with `lowest` read from a uniform buffer, which the compiler
can't see through; with those guards turned off, the random-circuit test fails on both drivers. Booleans are written
as `select(0, 1, score >= 0.5)`, since compilers fold `(score + K) - K`, and every literal is one exact f32. Not
guarded: GPUs may flush subnormal values (below 2^-126) to zero. Both GPUs tested here do: a subnormal operand reads as
0, and two values closer than 2^-126 compare equal. The tests check that every
comparison touching a subnormal value follows that rule, and every other one is exact. WGSL also lets drivers assume
no NaN or infinity occurs.

`cargo run --release -p handwired-rs-gpu --example search` searches 100,000 32-byte records for a 3-byte pattern with a
`#[handwired]` function (890 operations in 127 layers) and checks every result against the function. On an AMD Ryzen
laptop (Ryzen with 12 threads, integrated Radeon Renoir):

| | 100,000 records |
| --- | --- |
| the plain Rust function | 1.5 ms |
| CPU `TensorExecutor`, one record at a time | 660 ms |
| CPU `BatchExecutor`, 64 records at a time, one thread | 36 ms |
| CPU `BatchExecutor`, 12 threads | 13–15 ms |
| GPU, including upload and read-back | 30 ms |

`BatchExecutor` keeps each value as a block of 64 records, so every row is a loop the compiler vectorizes (with AVX2
when the CPU has it, picked at run time), and every lane repeats the one-record arithmetic in the same order: the
results are bit for bit those of `TensorExecutor`. `Compiled::run_batch` runs it on typed records across all cores,
encoding and range-checking them in parallel too; `encode_records` once and `run_encoded` / `gradients_encoded` many
times reuses records across tuning steps, and `broadcast(&mut encoded, arg, &value)` rewrites one argument, e.g. the
parameters a step changes, in every record (3.5 ms for 12 of 34 cells in 262,144 records, against 11–19 ms to encode
again).

`cargo run --release -p handwired-rs-gpu --example kv_store` is a fixed-capacity key-value store (16 slots, get, put,
delete) written as one `#[handwired]` function from the store's state to its next state and the answer. A `get` is hard
attention: a one-hot match of the key against every slot, multiplied into the values. The example verifies it against
the function, runs a put/get/delete session through the circuit, and runs 100,000 operations on 100,000 separate stores
in one GPU dispatch (1,082 operations in 70 layers; 50 ms on the GPU and 60 ms with `run_batch` on 12 threads,
against 6.6 ms for the plain function).

The circuit form doesn't make a program like this faster than native Rust; what it adds is that the same exact
program is a stack of sparse matrix multiplications with ReLUs, which runs on tensor hardware and is differentiable.
Set `WGPU_ADAPTER_NAME` (e.g. `llvmpipe`) to choose an adapter; the GPU tests skip when there is none.

## Embedded and elsewhere: `handwired-rs-run`

`compiled.export("name")` (or `TensorProgram::to_bytes`) writes a compiled program as bytes. The
[`handwired-rs-run`](run/README.md) crate runs them without handwired-rs: it is `no_std`, has no dependencies or
allocation, and builds for microcontrollers with and without an FPU. Business logic written as a `#[handwired]`
function and checked with `verify()` then ships to a device as a file:

- every run does the same operations whatever the inputs;
- results are bit-exact with the CPU executor wherever f32 arithmetic is IEEE 754;
- the file's hash says which logic ran.

Inputs outside the declared ranges, off their grids (e.g. 0.5 for an integer) or not an encoding of their type's values (an enum variant's cells set while another variant is selected), and damaged files, are refused. With its `std` feature it adds `handwired-run`, a
command that runs a program over records on standard input. A run costs about 1.3x the CPU executor. On an
STM32WL55 (Cortex-M4 without an FPU), every output of 8 test programs matched the host bit for bit; heater-style
logic took 1.7 ms at 32 MHz.

The checksum catches damaged files, not changed ones. [`handwired-rs-sign`](sign/README.md) adds Ed25519 signatures
as a file beside each program (`heater.hwc.sig`): a device checks that a program was signed by a key it trusts, for the
slot it is loaded into, with a release number it can use to refuse older programs. It is `no_std` and allocation-free,
with one dependency, and the program files and the runner don't change.

## Two larger examples

**SHA-1, ported from a library and checked against it.** `cargo run --release --example sha1` ports `sha1_smol`'s
block function (`Sha1State::process` and the intrinsic emulations it calls, function for function, under its BSD
license) to `#[handwired]`, on plain `u32` words with `wrapping_add`, `rotate_left` and `^ & |`, held as bytes inside
the circuit (see [Wide integers](#wide-integers)); bytes go in and out. The circuit has 68,036 operations in 1,616
layers (an earlier version with words as `[bool; 32]` had 99,698 in 4,011). It gives the standard test vectors'
digests, and the digests of 2,000 random messages of 0–300 bytes (6,061 blocks) match `sha1_smol`'s exactly: 371 ms
on the batch executor against 1.5 ms for `sha1_smol` itself, about 250x slower.

**Balancing a game through its rules.** `cargo run --release --example balance` writes a dice skirmish between two
units as one `#[handwired]` function (290 operations) and tunes three units' damage and health until every matchup of
Knight, Archer and Mage is 50/50. Each step counts the win rates exactly with the circuit and takes the direction from
the circuit's backward pass (the gradient of each fight's margin with respect to both units' stats), in a few
Gauss-Newton steps:

| | Knight beats Archer | Archer beats Mage | Mage beats Knight |
| --- | --- | --- | --- |
| before (6 dmg / 60 hp, 4 / 45, 9 / 36) | 97.8% | 28.6% | 22.4% |
| after 7 steps (6.57 / 39.6, 8.24 / 33.0, 7.35 / 56.5) | 50.3% | 49.5% | 50.8% |

The rates are counted on 20,000 fights per matchup that the tuning never saw. Gradients only help where the outcome
moves smoothly with the stats: in a first version, where units died mid-fight, most Archer–Mage fights ended with both
dead, a margin of exactly 0 with no gradient, and the tuning stalled at 58%. Win rates also move in steps of a few
points, since fights with the same hits change winner together, so about ±1% is as close as it gets.

## What's ported

| Area | Status |
| --- | --- |
| Four instructions (`affine`, `reglu`, `boolean`, `index_read`), stored-order f32 rounding | ✅ |
| Builder: term normalization, constant folding, small-sum composition, deduplication | ✅ |
| Declarations: `assume_range`, `assume_grid`, `assume_bit`, rounding points; range and grid inference | ✅ |
| `multiply`, `select`, `round_affine`, `emit_min/max`, `emit_and/or/not`, `emit_round` | ✅ |
| `emit_compare`: integer bits on grids, sign cascade, `end_with_boolean` | ✅ cost table matches handwired's docs |
| `emit_minimum_exact`, `emit_maximum_exact`, `emit_clamp_exact` | ✅ |
| Reference evaluator and reverse-mode backprop (boolean / index gradient rules) | ✅ |
| Layered sparse compiler with dead-operation removal, CPU executor | ✅ plus a vectorized, multithreaded batch executor |
| Source table cells (`input`, `event`, `state`, `state_bit`, grids, validation) | ✅ simplified |
| One-hot groups and gather lowering, `recompute_depths` | ✅ on the CPU executors and the GPU |
| `emit_argmax`, `emit_first_active` (selection trees that declare their one-hot groups) | ✅ costs match handwired's docs |
| Prefix networks (`sklansky_prefix`, `brent_kung_prefix`), and `dsl::running_sums` for `#[handwired]` | ✅ running-totals cost matches handwired's docs |
| `inline_circuit` (one circuit into another, sharing matching expressions) | ✅ |
| Springs (`emit_spring`, `emit_apply_spring_matrix`, `emit_animated_object`) and host timing (`timing`) | ✅ costs and planner logs match handwired's tests |
| `#[handwired]` proc-macro frontend (`macros/`, `src/dsl/`) | ✅ see above |
| GPU backend (`gpu/`: WGSL codegen, batch runner) | ✅ bit-exact on RADV and lavapipe |

One deliberate difference: cells declared as exact bits also get grid 1, so comparisons over sums of bits take the
cheap integer-bit form.

**One-hot groups and gathers.** `circuit.assume_one_hot(&bits, allow_all_zero)` declares that exactly one of `bits`
is 1 (or at most one). A sum of `multiply(bit, value)` over the whole group, e.g. the selected item's height, then
compiles to a gather: the executor finds the selected position once per group and reads that one value, instead of a
product per member. It's exact, up to the sign of an exactly zero result, for candidates with finite bounds; sums that
don't match the pattern exactly (a scaled term, a missing member, an extra term, two groups) stay dense.
`CompileOptions { recompute_depths: true, .. }` also moves what follows a gather up to the layer after its inputs.
`emit_argmax(circuit, scores, unit)` (the highest score in 0..1, ties to the left) and `emit_first_active(circuit, gate,
candidates)` (the first hit) build such selections as balanced trees, O(n) operations in O(log n) layers, and declare
the group themselves: `emit_argmax` always, `emit_first_active` only when the gate and candidates are proven bits
(`is_proven_bit`). At 64 candidates they're 437 operations in 33 layers and 316 in 31, as handwired's docs state.
`circuit.validate_one_hot_groups(&values)` checks the declaration in tests, since a false one makes the gather read
the wrong value. `#[handwired]` doesn't declare groups yet. For a read at a computed integer index, an index read (one
operation) is cheaper than a one-hot group; `#[handwired]` array reads already use one.

## Limitations

**What's exact, and where.** Every circuit computes exactly what its source computes on every input where the source
returns without panicking. Overflow and out-of-bounds indexing panic in Rust, so the circuit's value there is
unspecified. `verify()` samples inputs (plus each range's ends); it is a test, not a proof.

**The `#[handwired]` subset.**
- Not supported: `while`/`loop` without `#[max_iterations(N)]`, labeled loops, `break` with a value,
  generic structs and enums (other than `Option` and `Result`), closures, references, generic functions, `let` chains
  (`if let A = x && ...`), and calls to ordinary (non-`#[handwired]`) functions.
- Integer types wider than 2^24 (`u32`, `i32`, `u64`, ...) need declared `#[range]`s on parameters, and parameters,
  results and plain `+ - *` stay within ±2^24, the integers f32 holds exactly; shifts, bitwise operators, rotates,
  `wrapping_*` and `saturating_add`/`saturating_sub` go beyond as bytes. Divisors that may exceed 2^23 in magnitude
  and `f32` casts of values that may leave ±2^24 aren't supported; each fails the build with a message. There's no
  `saturating_mul`. `f32` parameters need `#[range]`s.
- An unannotated `let mut x = 0` defaults to `i32` where plain Rust would infer it from later use; annotate it.
- A lowercase single-name call `foo(x)` is taken to be another `#[handwired]` function (`foo_hw`). Iterate arrays by
  value (`for x in arr`), not with `.iter()`.
- An `if` with both branches ending in `return` needs an annotated binding when used as a value.
- Each `if` under an input-dependent condition emits both branches, so deeply nested `if`s and `match`es with early
  returns in many arms grow the generated code.
- Error messages for unsupported code come from rustc's type checker and can be indirect.

**Cost.** A circuit evaluates every path. Array writes at input-dependent indices cost one select per element, since
every element may change; early `return`s chain into deep layer stacks (the search example has 127 layers), and
integer division by a constant is a table read over the dividend's range (up to 65,536 entries; long division over
bytes beyond that), and division by an input value a table over both ranges or one comparison step per dividend bit. The search example
is still about 10x slower than plain Rust on 12 CPU threads, 20x on the GPU and 24x on one thread: the point is
exactness, differentiability and a tensor-shaped program, not speed.

**GPU.** Bit-exact against the CPU reference on AMD RADV (Renoir iGPU) and Mesa lavapipe; not tested on NVIDIA, Intel,
Apple or browsers. Both tested GPUs flush subnormal values to zero, so a subnormal operand reads as 0 and values
closer than 2^-126 compare equal, matching handwired's `flushSubnormals` executor; anything else is exact. WGSL
lets drivers assume there are no NaNs or infinities. There's no backward pass on the GPU. A program longer than 2,000
statements runs as stages, which pass values through memory and run slower than one shader would: about 2x for the
wide hash on an integrated GPU, and more where hundreds of bits cross each cut.

**Differentiability.** The reference evaluator has an exact backward pass, on the CPU only.
`compiled.gradients(&records, output, weights, options)` sums it over many records on every core, and the `balance`
example tunes a game with it. Comparisons, boolean atoms and index reads are step functions: their inputs get no
useful gradient, and by default a nonzero gradient reaching a boolean atom or an index is an error rather than a
silent zero (`BackpropOptions` picks straight-through or zero). A value read at a computed index, e.g.
`params[faction * 2 + stat]`, does get the gradient, on the entry the index selects.

**Gradients through comparisons.** `BackpropOptions { comparison_gradient: ComparisonGradient::Surrogate { width },
.. }` gives comparisons (and so `if`s on inputs, `max`, `min`, `clamp`) a gradient anyway: the backward pass treats
each comparison bit as `sigmoid(d / width)` of its difference `d`, e.g. `a - b`, and sends that slope straight to the
operands, while the forward values stay exact. `RelativeSurrogate { fraction }` sets each comparison's width to
`fraction` × its own difference range instead, for circuits that mix scales, e.g. dice and hit points. It's what lets
a threshold be tuned at all, e.g. "hits on a roll of at least t", whose exact gradient is zero everywhere. There's no
export to a deep learning framework.

**Not ported from handwired.** Index-read tables stored as a base and a stride, source-table cell declarations beyond
the simplified ones, and its demos (HandwiredOS, the learned gallery, JSON search). The springs' end-to-end
frame-trace tests (whole animations along 10–120 fps traces) aren't ported; their parts are.

## License

MIT. handwired is MIT-licensed, Copyright (c) 2026 Cheng Lou; this port keeps that notice in [LICENSE](LICENSE).

## Tests

`cargo test` runs:

- `tests/compare.rs`: the operation/layer counts of every comparison construction against the table in handwired's
  `docs/execution-model.md`, and every bit against the native f32 comparison on equal values, one-ulp neighbors,
  subnormal differences and range ends, through both evaluators.
- `tests/tensor.rs`: seeded random circuits, where the compiled program has to reproduce every live value of the
  reference evaluator, and the batch executor has to reproduce the one-record executor bit for bit, in full and
  partial batches, on one thread and several.
- `tests/gather.rs`: gather lowering, ported from handwired's gather-lowering tests: which masked sums lower (both
  product orientations, mixed-sign products, biases, chained gathers, groups that allow all zeros) and which stay dense
  (scaled, missing or extra terms, two groups, values that may be infinite), one index resolution per group, extreme
  values, recomputed depths, a violated declaration, and seeded random circuits with hand-declared, `emit_argmax` and
  `emit_first_active` groups, through both CPU executors.
- `tests/select.rs`: `emit_argmax` and `emit_first_active`, ported from handwired's tests: winners, ties to the left,
  the non-transitive tie band, which selections declare a group, a fractional selection that must not lower, and
  their costs at 64 candidates against handwired's docs.
- `tests/prefix.rs`: prefix networks, ported from handwired's tests: every prefix in order for 1 to 70 items, the
  combine and depth formulas, a state machine scan against its loop, the running-totals cost at 1,024 rows against
  handwired's docs, and `dsl::running_sums` in `#[handwired]` functions.
- `tests/inline.rs`: `inline_circuit`, ported from handwired's tests (shared expressions, kept rounding and term
  order, declarations on read wires, exact bits, one-hot groups), and random circuits inlined into a larger one with
  every value kept. A plain `affine`'s terms are summed in wire order, so inlining through permuted cells can change
  its last bit; `round_affine` keeps its order.
- `tests/springs.rs`: springs and host timing, ported from handwired's spring-matrix tests: a spring matrix against
  float64 Euler steps and an FP32 twin, its cost (14 operations in 3 layers), animated objects' costs and an
  independent FP32 twin bit for bit (settle thresholds, exact rest, subnormals, snaps), rebasing, refusals, inlining,
  gradients, frame timing, step counts, matrix powers and bounds, and `advance_frame`'s exact plans (settle checks,
  snaps past the cap, late events, write-back policies, frame-rate independence), plus `emit_spring` settling.
- `tests/backprop.rs`: gradients of products, selects, boolean atoms and index reads.
- `tests/helpers.rs`: `emit_round` (ties to even), `emit_or`, `emit_clamp_exact` and `emit_maximum`.
- `tests/macro_*.rs`: `#[handwired]` functions (the README example, the toggle, histograms, table lookups, nested
  array writes, `match`, wrapping arithmetic, casts, order-sensitive f32 sums, early `return`/`break`/`continue`
  in nested loops), each verified against its own function.
- `tests/toggle.rs`: handwired's toggle-switch template (without its spring), compared frame by frame with the same
  switch in plain Rust, over a scripted trace and random traces near the track edges.

- `gpu/tests/exact.rs`: random circuits, comparisons at their edges, gathers over one-hot groups, selection trees and
  `#[handwired]` batches on the GPU, compared bit for bit with the CPU reference (skipped without an adapter), each run
  with the default stages and again split every 5 statements, so every kind of operation reads values carried from
  earlier stages; range checks on batches, and comparisons of subnormal values, checked against the
  flushing model when the GPU flushes them.

`cargo run --example toggle` builds the switch and prints its size. `cargo run --release --example ui_panel` runs a
toggle and a slider (pointer handling, arming, dragging with clamping and snapping, and an easing animation: 262
operations) through a scripted pointer, and `cargo run --release --example life` runs Conway's Game of Life on a
24 × 24 torus (6,912 operations in 8 layers); both check every frame against their function and print the frames.

CI (`.github/workflows/ci.yml`) runs all of the above on GitHub's Ubuntu runners, in debug and release builds, with the
GPU tests on Mesa's lavapipe software Vulkan (it fails rather than skips if no adapter appears), runs every example,
and checks `cargo fmt`, `clippy -D warnings` on all targets, and packaging of the macro crate.
