# Changelog

## 0.2.0 (handwired-rs, handwired-rs-gpu)

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
