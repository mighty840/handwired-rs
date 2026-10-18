# Checking handwired-rs-run on hardware

- `suite/` is a small `no_std` crate. It runs a suite of programs and compares every output bit with the outputs
  handwired-rs's `TensorExecutor` computed on the host.
- `firmware/` embeds a suite and prints one PASS/FAIL line per program, with cycles per run from the DWT counter
  where there is one. The last line is `=== N passed, M failed ===`.

Write a suite (60 random records per program plus each range's ends; `--small` leaves out the 84 KB wide hash):

```sh
cargo run --release -p handwired-rs-run --example hardware_suite -- run/hardware/firmware/suite.bin 60
```

Then build the firmware from `run/hardware/firmware`:

| Board | Build | Output |
|---|---|---|
| NUCLEO-WL55JC1 (Cortex-M4, no FPU) | `cargo build --release --features wl55 --target thumbv7em-none-eabi` | LPUART1 (ST-LINK VCP), 115200 |
| QEMU lm3s6965evb (Cortex-M3) | `cargo build --release --features qemu --target thumbv7m-none-eabi` | semihosting |
| QEMU microbit (Cortex-M0) | `cargo build --release --features qemu --target thumbv6m-none-eabi` | semihosting, no cycle counts |

Run it on QEMU:

```sh
qemu-system-arm -cpu cortex-m3 -machine lm3s6965evb -nographic -semihosting-config enable=on,target=native \
  -kernel target/thumbv7m-none-eabi/release/handwired-run-check
qemu-system-arm -machine microbit -nographic -semihosting-config enable=on,target=native \
  -kernel target/thumbv6m-none-eabi/release/handwired-run-check
```

CI runs both QEMU machines. QEMU runs the real ARM instructions, including Rust's software floating point, but its
cycle counts mean nothing; those come from hardware.
