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
- CI runs clippy on all targets and the tests in release builds too.

`handwired-rs-macros` is unchanged at 0.1.0.

## 0.1.0

First release.
