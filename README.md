# handwired-rs

[![CI](https://github.com/mighty840/handwired-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/mighty840/handwired-rs/actions/workflows/ci.yml)

A Rust port of [chenglou/handwired](https://github.com/chenglou/handwired) by Cheng Lou, plus a `#[handwired]` attribute
that compiles ordinary Rust functions into handwired circuits, and a WebGPU backend.

A handwired program is a fixed list of f32 operations: weighted sums, ReLU products, a rounding-to-bit atom and table
reads. It runs as layers of sparse matrix multiplications, the shape of a neural net, and stays differentiable, while
computing exactly what the ordinary code computes. The ideas, the instruction set, the exactness rules and the
comparison constructions are handwired's; this crate ports them to Rust and checks itself against handwired's
published numbers.

**Status:** 0.1, experimental. Written in one session with Claude Code and checked by the tests listed below, on one
machine. Not reviewed by handwired's author. Read [Limitations](#limitations) before relying on it.

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
```

The attribute keeps the function as it is, as the oracle, and adds `toggle_step_hw` (the body over circuit values,
which other `#[handwired]` functions call) and `toggle_step_circuit()` (builds and compiles it). The body runs once at
build time with every input-dependent value as a typed `Val<T>`; operators on `Val`s append operations, so ordinary
code like `a + b` or `x.min(y)` stays as written. The macro rewrites only what Rust can't overload: comparisons,
`&&`/`||`, indexing, assignment, `if`, `match` and `as`.

**The contract:** the circuit agrees with the function on every input where the function returns without panicking.
Overflow and out-of-bounds indexing panic in the oracle, so their circuit values are unspecified. `verify` skips
inputs where the oracle panics; run it with overflow checks on (e.g. a debug build).

**Supported:** `bool`, `u8`–`u64`, `i8`–`i64`, `usize`, `f32`, arrays and tuples as parameters, results and locals;
`+ - * !`, comparisons, `&& || & | ^`; integer `/` and `%` by a constant (as a table read); f32 `/` by a power of two;
`min max clamp abs wrapping_add/sub saturating_add/sub`; `as` casts between them (f32 → int truncates and saturates
like Rust); `if`/`else` as a statement or value; `match` with literals, ranges, `|`, guards, tuples, bindings and
consts; `for` over ranges or arrays that don't depend on the inputs (they unroll); array reads and writes at computed
indices, including nested arrays; calls to other `#[handwired]` functions; `return`, `break` and `continue` anywhere,
including under conditions that depend on the inputs.

**Early exits:** the trace runs every path, so an exit can't leave it. A function that returns early gets a `done` bit
and a result slot: each `return` records its value where no earlier return happened, and the function's value picks
the recorded one. A loop with `break`/`continue` gets a `halted` bit, and assignments inside it apply only where it
still runs. Under native conditions, e.g. `if index == 3 { break; }`, exits leave the unrolled loop for real, and
selects on flags that no input-dependent branch has set fold away, so they cost nothing. A `return` outside input-dependent
branches and flagged loops also leaves for real.

**Declarations:** `#[range(lo, hi)]` is required on `f32` parameters and on integer types wider than 2^24 (`u32`,
`i32`, ...), whose values must stay within ±2^24 (the build panics, pointing at the operation, if they may not).
`#[grid(step)]` declares a power-of-two grid, e.g. `1.0 / 64.0` for pointer coordinates, which makes comparisons
cheap.

See [Limitations](#limitations) for what isn't supported.

## GPU: `handwired-rs-gpu`

The `gpu/` crate runs a compiled program as one WebGPU compute shader (wgpu 25), one invocation per record, so a
batch of records runs in parallel. It reproduces the CPU reference evaluator bit for bit.

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
guarded: GPUs may flush subnormals to zero, which would change a comparison of values closer than 2^-126 (untested
here), and WGSL lets drivers assume no NaN or infinity occurs.

`cargo run --release -p handwired-rs-gpu --example search` searches 100,000 32-byte records for a 3-byte pattern with a
`#[handwired]` function (890 operations in 127 layers) and checks every result against the function. On an AMD Ryzen
laptop's integrated Radeon (Renoir):

| | 100,000 records |
| --- | --- |
| the plain Rust function | 1.7 ms |
| CPU tensor executor, one thread | 754 ms |
| GPU, including upload and read-back | 27 ms |

The circuit form doesn't make a program like this faster than native Rust; what it adds is that the same exact
program is a stack of sparse matrix multiplications with ReLUs, which runs on tensor hardware and is differentiable.
Set `WGPU_ADAPTER_NAME` (e.g. `llvmpipe`) to choose an adapter; the GPU tests skip when there is none.

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
| Layered sparse compiler with dead-operation removal, CPU executor | ✅ |
| Source table cells (`input`, `event`, `state`, `state_bit`, grids, validation) | ✅ simplified |
| One-hot groups and gather lowering, `recompute_depths` | ⏳ not yet |
| Springs / host timing, `emit_argmax`, `emit_first_active`, `inline_circuit`, prefix networks | ⏳ not yet |
| `#[handwired]` proc-macro frontend (`macros/`, `src/dsl/`) | ✅ see above |
| GPU backend (`gpu/`: WGSL codegen, batch runner) | ✅ bit-exact on RADV and lavapipe |

One deliberate difference: cells declared as exact bits also get grid 1, so comparisons over sums of bits take the
cheap integer-bit form.

## Limitations

**What's exact, and where.** Every circuit computes exactly what its source computes on every input where the source
returns without panicking. Overflow and out-of-bounds indexing panic in Rust, so the circuit's value there is
unspecified. `verify()` samples inputs (plus each range's ends); it is a test, not a proof.

**The `#[handwired]` subset.**
- Not supported: `while`/`loop` (every loop needs bounds that don't depend on the inputs), labeled `break`/`continue`,
  division or remainder by an input value, shifts and integer bitwise operators, structs, enums, `Option`, closures,
  references, generic functions, `if let`, and calls to ordinary (non-`#[handwired]`) functions.
- Integer types wider than 2^24 (`u32`, `i32`, `u64`, ...) need declared `#[range]`s, and every value must stay within
  ±2^24, the integers f32 holds exactly. `f32` parameters need `#[range]`s.
- An unannotated `let mut x = 0` defaults to `i32` where plain Rust would infer it from later use; annotate it.
- A lowercase single-name call `foo(x)` is taken to be another `#[handwired]` function (`foo_hw`). Iterate arrays by
  value (`for x in arr`), not with `.iter()`.
- An `if` with both branches ending in `return` needs an annotated binding when used as a value.
- Each `if` under an input-dependent condition emits both branches, so deeply nested `if`s and `match`es with early
  returns in many arms grow the generated code.
- Error messages for unsupported code come from rustc's type checker and can be indirect.

**Cost.** A circuit evaluates every path. Array writes at input-dependent indices cost one select per element (gather
lowering isn't ported), early `return`s chain into deep layer stacks (the search example has 127 layers), and
integer division by a constant is a table read over the dividend's range (at most 65,536 entries). On the CPU, a
circuit is far slower than the function it came from, and on the GPU the search example is still about 16x slower
than plain Rust: the point is exactness, differentiability and a tensor-shaped program, not speed.

**GPU.** Bit-exact against the CPU reference on AMD RADV (Renoir iGPU) and Mesa lavapipe; not tested on NVIDIA, Intel,
Apple or browsers. Subnormal flushing isn't tested, and would change comparisons of values closer than 2^-126. WGSL
lets drivers assume there are no NaNs or infinities. There's no backward pass on the GPU.

**Differentiability.** The reference evaluator has a backward pass, tested on small cases; nothing here trains a
circuit yet, and there's no export to a deep learning framework.

**Not ported from handwired.** One-hot groups and gather lowering, depth recomputation, springs and frame timing,
`argmax`/`first_active`, circuit inlining, prefix networks, and its demos (HandwiredOS, the learned gallery, JSON
search).

## License

MIT. handwired is MIT-licensed, Copyright (c) 2026 Cheng Lou; this port keeps that notice in [LICENSE](LICENSE).

## Tests

`cargo test` runs:

- `tests/compare.rs`: the operation/layer counts of every comparison construction against the table in handwired's
  `docs/execution-model.md`, and every bit against the native f32 comparison on equal values, one-ulp neighbors,
  subnormal differences and range ends, through both evaluators.
- `tests/tensor.rs`: seeded random circuits, where the compiled program has to reproduce every live value of the
  reference evaluator.
- `tests/backprop.rs`: gradients of products, selects, boolean atoms and index reads.
- `tests/helpers.rs`: `emit_round` (ties to even), `emit_or`, `emit_clamp_exact` and `emit_maximum`.
- `tests/macro_*.rs`: `#[handwired]` functions (the README example, the toggle, histograms, table lookups, nested
  array writes, `match`, wrapping arithmetic, casts, order-sensitive f32 sums, early `return`/`break`/`continue`
  in nested loops), each verified against its own function.
- `tests/toggle.rs`: handwired's toggle-switch template (without its spring), compared frame by frame with the same
  switch in plain Rust, over a scripted trace and random traces near the track edges.

- `gpu/tests/exact.rs`: random circuits, comparisons at their edges and `#[handwired]` batches on the GPU, compared bit
  for bit with the CPU reference (skipped without an adapter).

`cargo run --example toggle` builds the switch and prints its size.

CI (`.github/workflows/ci.yml`) runs all of the above on GitHub's Ubuntu runners, with the GPU tests on Mesa's lavapipe
software Vulkan (it fails rather than skips if no adapter appears), runs the search example, and checks `cargo fmt`,
`clippy -D warnings` on the libraries, and packaging of the macro crate.
