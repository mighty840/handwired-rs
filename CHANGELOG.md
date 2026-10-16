# Changelog

## Unreleased

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
