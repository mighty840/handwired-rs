# Changelog

## Unreleased

- handwired-rs-run takes the same time whatever the values, as far as the chip allows:
  - ReLU and table reads use masks instead of branches on the data;
  - table reads and the input range check compare in f32, so FPUs without f64 don't call software routines whose
    time depends on the values.
  On an STM32H563 the slowest record of each benchmark took 3 to 24 cycles longer than its fastest (at most 0.03%),
  down from 10 to 116. Results are unchanged; tables over 2^24 entries are refused at load.
- meta-handwired, a Yocto/OpenEmbedded layer (scarthgap): a `handwired-run` recipe (also native and nativesdk),
  `handwired-program.bbclass`, which installs `.hwc` programs under `/usr/share/handwired` with a sha256 manifest
  after checking each at build time, an example program, and runtime tests for `testimage`. CI builds
  `core-image-minimal` and tests it under QEMU. handwired-rs-run supports Rust 1.75 (edition 2021), and the workspace
  uses resolver 2 and a version 3 lockfile so its packages stay readable by older Cargo.
- New crate handwired-rs-run 0.1.0: runs exported programs without handwired-rs. It is `no_std`, with no allocation,
  dependencies or `unsafe`, and builds for Cortex-M0, Cortex-M4F/M7 and RV32IMC. It is bit-exact with
  `TensorExecutor` on every value. It refuses inputs outside the declared ranges, and refuses damaged bytes at load
  (checksum, then structure), so a run never reads out of bounds. With `std`, the `handwired-run` command runs a
  program over records on standard input.
- `Compiled::export(name)` and `TensorProgram::to_bytes(description)` write programs in that format (version 1,
  described in `handwired_rs::export`): the source cells' declared ranges, the outputs' positions and the layers,
  with a checksum.
- `saturating_add` and `saturating_sub` on values held as bytes, and on one-wire values whose result may leave
  ±2^24: the wrapping result in bytes, or the limit it passed where it wrapped, found by comparing it with the left
  operand. They used to fail the build. Narrow circuits are unchanged.
- Bit splits declare their ReLU ramps' ranges and integer grids: each step knows what's left is below twice its
  threshold, which the sum of its terms' ranges doesn't show. Circuits are unchanged, but every value in bit-heavy
  programs now has a known integer range, so GPU stages pack all of them: the wide hash carries 22 words per record
  between stages instead of 87 (its run time is unchanged), and `TensorProgram::ranges`/`grids` are tighter.

## 0.8.0 (2026-10-18): handwired-rs and handwired-rs-gpu 0.8.0, handwired-rs-macros 0.5.0

- `<`, `<=`, `>`, `>=`, `min`, `max` and `clamp` on `Option`, `Result` and enums with data, in the order Rust derives:
  the variant in declaration order, then its fields in order. Comparing two native values of such an enum no longer
  panics and needs no `PartialOrd`. `min`, `max` and `clamp` also work on `bool`s and fieldless enums. `dsl` gains
  `cells_compare` and `compare_encoded`; the native `Option` and `Result` comparisons need `T: HwType`.
- `Result<T, E>`, laid out and used like `Option`: `Ok(..)`/`Err(..)`, patterns, `==`/`!=`, `is_ok`, `is_err`, `ok`,
  `err`, `unwrap_or`, `unwrap`, `expect`, `unwrap_err` and `expect_err`. `?` on a `Result` or an `Option` in a function
  that returns one, leaving early like `return` (no `From` conversion of the error). `dsl` gains `ResultHw`, `ok` and
  `err`.
- A field of a struct or variant literal takes its type from the field, so plain integer literals in an `if`,
  `match`, block, tuple or array there work: `Action::Attack(if strong { 6 } else { 3 })` no longer makes an `i32`.
  Circuits that compiled before are unchanged. `dsl` gains `choose`.
- `Option<T>` and `#[derive(HwType)]` enums whose variants carry data, as parameters, results, locals, struct fields
  and array elements: one cell for the variant, then every variant's fields, which hold 0 where the value is another
  variant (their declared ranges take in 0). Constructors (`Action::Attack(d)`, `Action::Move { dx, dy }`,
  `Some(x)`), patterns that bind fields by position or name and nest, `==`/`!=`, and `is_some`, `is_none`,
  `unwrap_or`, `unwrap` and `expect` on Options. The derive adds `<Name>Hw` for such enums. `Attrs` gains
  `holds_zero` (breaking for code that builds `Attrs` with a struct literal), and `dsl` gains `OptionHw`, `some`,
  `Native` and `cells_equal`. Needs handwired-rs-macros 0.5.0.
- `/` and `%` by a value that depends on the inputs: one table read of every pair's result where the two ranges have
  65,536 pairs or fewer, else long division over the dividend's bits, one comparison step per bit, for divisors within
  ±2^23 and dividends of any width, signed or not. `10 / x` and `10 % x` with a literal on the left work too.
- Fixed: `x op= a op b` applied `op` to `x op a` and then `b`, e.g. `score -= a - b` computed `score - a - b`, and
  f32 `y += f + g` rounded as `(y + f) + g`. The right side is now grouped as in Rust. Circuits for such statements
  change (`+=` of a sum of integers compiles as before).

## 0.7.0 (2026-10-17): handwired-rs and handwired-rs-gpu 0.7.0

- GPU stages pass integers within small ranges as only the bits their ranges need, packed into 32-bit words (the
  carried buffer is now `array<u32>`; other values pass as their f32 bits), and cuts count values in words: the wide
  hash carries 87 words per record instead of 152 and runs roughly 30% faster. `TensorProgram` gains `ranges` and
  `grids`, each packed position's range and grid in the circuit. A packed -0 comes back as +0, within the GPU's
  existing "up to the sign of zero" match.
- `/` and `%` by a constant on wide values, and on one-wire values of wide types whose range is too wide for a table
  (65,536 entries or more): long division over the bytes, a byte at a time for divisors up to 65,536 and fewer bits
  at a time up to 2^23 (larger divisors fail the build), truncating toward zero like Rust; unsigned division by a
  power of two is a shift and a mask. The remainder comes back as one wire. `abs` on wide values. Signed right shifts
  of wide values, e.g. `x >> 24` on an `i32`, get one wire too.
- Wide values narrow again: bytes whose high bytes are known 0s (or known 255s, for a negative value of a signed type),
  e.g. after `h >> 24` or `h & 0xFFFF`, also get one wire within ±2^24, so they can be returned, compared, cast to
  `f32` or used in plain arithmetic without a cast, and keep their bytes for bitwise operators and shifts; the form
  nothing reads is removed at compile time. One-wire arithmetic on such a value that may leave ±2^24 goes back to
  bytes instead of failing the build. `Val::is_wide()` now means held only as bytes.
- GPU: a program longer than `wgsl::WHOLE_STATEMENTS` (2,000) statements is split into stages, separate shaders that
  run one after another and pass the values later stages read through a storage buffer: `wgsl::stage_statements`
  gives 1,000 statements per stage up to 20,000, shrinking with the square root of the length to 250 from 320,000.
  Drivers' compile time grows much faster than a shader's length, so a 1,131-operation wide hash sets up in 0.47 s
  instead of 16–27 s on RADV, and SHA-1 (68,036 operations) in 8–17 s; staged programs run slower than one shader would
  (about 2x for the wide hash). Cuts go where the fewest values cross per statement, stages compile on every core,
  and a run splits into several dispatches if the carried values would pass the device's storage binding limit.
  `wgsl::emit_staged`, `GpuProgram::from_program_staged` and `GpuBatch::staged` take another stage size, and
  `GpuProgram::dispatch_records` caps the records per dispatch.
- `GpuProgram::wgsl` and `wgsl::Shader::code` are replaced by `stages: Vec<String>`, and shaders bind a fifth buffer
  for carried values (breaking for code that read the WGSL or bound it itself).

## 0.6.0 (2026-10-17): handwired-rs and handwired-rs-gpu 0.6.0, handwired-rs-macros 0.4.0

- Wide integers: `u32`, `i32`, `u64`, `i64` and `usize` values may go beyond ±2^24 inside a circuit, held as bytes,
  where shifts, bitwise operators, rotates, `wrapping_add/sub/mul`, constants beyond ±2^24 or already wide operands
  take them. Byte arithmetic carries from byte to byte (multiplication adds columns of byte products), comparisons
  are the carry of `a + !b + 1`, and wide values go through casts, `if`s, `match`es and table reads. Parameters and
  results stay one cell within ±2^24 (returning a wide value fails the build with a message), and plain `+ - *` of
  one-wire values keep their ±2^24 check. New: `rotate_left`, `rotate_right` and `wrapping_mul` on circuit integers,
  for every integer type. The SHA-1 example now uses `u32` words: 68,036 operations in 1,616 layers (was 99,698 in
  4,011 with `[bool; 32]` words), about 2x faster.
- `saturating_add`/`saturating_sub` on a wide value fail the build instead of wrapping first and clamping after.
- `Val`'s wire is now read with `Val::wire()`, which panics for a wide value (breaking for code that read the field);
  `Val::is_wide()` tells them apart.
- Integer `if` merges keep their result within the type's range, so loop-carried integers no longer widen every pass.
- `while`, `while let` and `loop` in `#[handwired]` functions with `#[max_iterations(N)]`: the circuit runs a `for`
  loop over `0..N` that breaks where the loop would end; the function counts its passes and panics via
  `dsl::iteration_limit` if it would start pass N + 1, which `verify` reports as the new
  `VerifyError::IterationLimit` instead of skipping. `VerifyError` is now `#[non_exhaustive]` (breaking for
  exhaustive matches).
- `let (mut a, mut b) = (x, 0)` in `#[handwired]` functions: each `mut` name in a tuple pattern starts as a circuit
  value, like a plain `let mut`, so loops and branches can assign it.

## 0.5.0 (2026-10-16): handwired-rs and handwired-rs-gpu 0.5.0, handwired-rs-macros 0.3.0

- Integer bitwise operators in `#[handwired]` functions: `&`, `|`, `^`, `!`, `<<`, `>>`, their `op=` forms and
  `count_ones()`, on values within ±2^24. A value is split into two's-complement bits, one layer per bit up to 11 bits
  and two above; AND is one ReLU per bit, and OR and XOR are sums over it. Shifts by a constant move bits (`<<` wraps like Rust only where the
  result may leave the type), shifts by an input value read the result for each amount. A u8 XOR is 40 operations in
  10 layers. `dsl::shl`, `dsl::shr` and `dsl::Shift` are what the macro calls.
- `#[derive(HwType)]` on enums whose variants carry no data (handwired-rs-macros): the enum is a scalar, its
  discriminant in one cell, declared as the discriminants' range and sampled only as real variants. `match` on it,
  `==`/`!=`, ordering with a derived `PartialOrd`, `as` casts to integers, enum fields, arrays of enums at computed
  indices and enum results work in `#[handwired]` functions.
- `matches!(x, pattern if guard)` and `if let pattern = x { .. } else { .. }` in `#[handwired]` functions, lowered
  like `match`, so they work on enums, integers, tuples and bindings.
- `Scalar` no longer requires `PartialOrd` (breaking for generic code that compares `T: Scalar` values with `<`), and
  gains `default_value()` (an enum's first variant). `dsl::declare_scalar`,
  `dsl::sample_variant` and `dsl::compare_native` are public for the derive.

## 0.4.0 (2026-10-16): handwired-rs and handwired-rs-gpu 0.4.0 (handwired-rs-macros stays 0.2.0)

- `verify`, `verify_seeded` and `verify_with` check their samples on every core: one `verify(20,000)` of a
  1,053-operation circuit in a debug build goes from 2.6 s to 0.54 s. Inputs are still drawn in order from the seed,
  and the first failure in sample order is reported. They now need `Args: Sync`, so generic callers add that bound.
- `Compiled::broadcast(&mut encoded, arg, &value)`: one argument, e.g. a tuning step's parameters, written into every
  encoded record and range-checked once; `Params::arg_cells(index)` gives an argument's cells.
- `ComparisonGradient::RelativeSurrogate { fraction }`: each comparison's width is `fraction` × the span of its
  difference's declared range, so dice and hit-point thresholds in one circuit each get a fitting width.
- `run_batch` encodes and range-checks records in parallel (about 1.8x faster end to end on 262,144 records), and
  `Compiled::encode_records` / `run_encoded` / `gradients_encoded` reuse encoded records across steps. `run_batch`
  now needs `Args: Sync` and `R: Send`.
- `ComparisonGradient::Surrogate { width }` in `BackpropOptions`: comparison bits get the gradient of
  `sigmoid(d / width)` of their difference in the backward pass, straight to their operands, so thresholds, `if`s on
  inputs and `max`/`min`/`clamp` can be tuned; the forward values stay exact. Operations record their comparison
  (`Operation::comparison`), and inlining carries it over. Breaking: `BackpropOptions` gains a field, so build it with
  `..Default::default()`.

## 0.3.0 (2026-10-16): handwired-rs and handwired-rs-gpu 0.3.0, handwired-rs-macros 0.2.0

- Springs and host timing, ported from handwired: `emit_spring`, `emit_apply_spring_matrix` and
  `emit_animated_object` (springs stepped by a host-written matrix and settled together with one exact bit), and the
  `timing` module (`quantize_frame_timing`, `count_spring_steps`, spring matrices and bounds, `advance_frame`). Costs
  and planner logs match handwired's tests.
- `Circuit::inline_circuit`, ported from handwired: rebuilds one circuit's operations in another, through the
  builder, so matching expressions are shared; declarations, rounding points, comparison bits and one-hot groups
  carry over.
- Prefix networks, ported from handwired: `sklansky_prefix` and `brent_kung_prefix` for any associative combine. At
  1,024 rows, Brent-Kung running totals are 2,036 operations in 18 layers, as handwired's docs state.
- `Compiled::gradients(&records, output, weights, options)`: the exact backward pass summed over many records with
  per-record weights, split across all cores. The `balance` example uses it and runs about 2x faster.
- `Compiled::verify_with(cases, seed, sampler)` checks inputs from a custom sampler, e.g. reachable game states.
  `verify` now returns a typed `dsl::VerifyError` (`Mismatch` or `OutOfRange`) instead of a `String`.
- A test that gradients through a read at a computed index, e.g. `params[faction * 2 + stat]`, reach the selected
  entry.
- `#[derive(HwType)]` (handwired-rs-macros 0.2.0): structs with named fields as `#[handwired]` parameters, locals and
  results, with `#[range]`, `#[ranges]` and `#[grid]` per field; field reads and writes, also through computed indices,
  struct literals with `..base`, and structs as an `if`'s value. It adds `<Name>Hw`, the struct's circuit form.
- `#[ranges([(lo, hi), ...])]` or `#[ranges(CONST)]` on a parameter: one range per scalar. `Attrs` gains `ranges`.
- Tuples of up to 12 and parameter lists of up to 16.
- `<fn>_hw` and `<fn>_circuit()` get doc comments and the function's lint attributes, so `missing_docs` passes.
- `handwired_rs::dsl::running_sums` for `#[handwired]` functions: running totals of an integer array, a loop in the
  function and a Brent-Kung network in the circuit.

## 0.2.0 (2026-10-16): handwired-rs, handwired-rs-gpu

- `BatchExecutor`: runs a compiled program on 64 records at a time, vectorized (AVX2 when available, picked at run
  time) and bit for bit identical to `TensorExecutor`; `BatchExecutor::run_parallel` splits records across threads.
  On the search example, 100,000 records take 36 ms on one thread and 13–15 ms on 12, down from 660 ms.
- `Compiled::run_batch`: typed records in, typed results out, range-checked, on every core.
- One-hot groups and gather lowering, ported from handwired: `Circuit::assume_one_hot` declares a group, a masked sum
  over it compiles to a read of the selected candidate on the CPU executors and the GPU, and
  `Circuit::validate_one_hot_groups` checks the declaration. `Circuit::compile_with` takes `CompileOptions`
  (`lower_gathers`, on by default, and `recompute_depths`). `TensorLayer` and `TensorProgram` gain gather fields.
- GPU: a test of comparisons on subnormal values. Both tested GPUs (RADV, lavapipe) flush subnormals to zero; every
  comparison touching one follows that model, and every other comparison stays exact. Documented in the README.
- `emit_argmax` and `emit_first_active`, ported from handwired, with `is_proven_bit`: balanced selection trees that
  declare their one-hot groups, so masked reads of their selections lower to gathers. Their costs at 64 candidates
  match handwired's docs.
- Examples: `sha1` ports sha1_smol's SHA-1 block function and checks every digest against it; `balance` tunes a
  dice game's unit stats to 50/50 matchups with gradients from the circuit's backward pass.
- `GpuBatch::run` and `GpuBatch::encode` check every record against the declared ranges, as the docs said; `encode`
  now returns a `Result`.
- CI runs clippy on all targets and the tests in release builds too.

`handwired-rs-macros` is unchanged at 0.1.0.

## 0.1.0

First release.
