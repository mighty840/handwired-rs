# handwired-rs-sign

Signatures for program files exported by [handwired-rs](https://crates.io/crates/handwired-rs). A device checks
that a program comes from a key it trusts, for the slot it is loaded into, before handing the bytes to
[handwired-rs-run](../run/README.md). `no_std`, no allocation, no `unsafe`; its one dependency,
[`ed25519-compact`](https://crates.io/crates/ed25519-compact), has none of its own.

The runner's checksum catches damaged files, not changed ones: anyone who can write a file can write a valid
checksum. A signature can only be made with the secret key, which stays with whoever releases programs, never on the
device or in the image.

- **Next to the program, not inside it.** `heater.hwc.sig` sits beside `heater.hwc`, whose bytes, and the SHA-256
  that identifies them in manifests and logs, don't change. Runners that don't check signatures read the same files.
- **For one slot.** A signature names the slot the program is for (`estop`, `heater`), so a validly signed program
  can't be loaded in place of another.
- **With a release number.** A signature carries a sequence number. [`verify`] returns it, and the device must refuse
  one below the last it accepted for that slot, or an old, validly signed program could be loaded again. Keeping
  that number, and a log of what was loaded, is the device's job.

It is Ed25519 (RFC 8032) over `"handwired-rs signed program v1\0" ‖ slot length (1 byte) ‖ slot ‖ sequence (u64
LE) ‖ program`. Verification reads the program once and needs no buffer for it.

## Release

```sh
cargo install handwired-rs-sign --features std
handwired-sign keygen release.key release.pub      # once; keep release.key offline
handwired-sign sign --key release.key --slot heater --sequence 4 heater.hwc   # writes heater.hwc.sig
handwired-sign verify --key release.pub heater.hwc  # heater.hwc: slot heater, sequence 4, signed by …
```

Keys are raw files: a secret key is 64 bytes (seed, then public key), a public key 32. `keygen` reads its seed from
`/dev/urandom`, writes the secret key readable by its owner only, and never replaces a file. Signing is deterministic,
so the same program, slot, sequence and key always give the same signature.

## Device

```rust,ignore
#![no_std]
use handwired_rs_run::Program;

const TRUSTED: [[u8; 32]; 1] = [*include_bytes!("release.pub")];

fn load<'a>(program: &'a [u8], signature: &[u8], last_sequence: u64) -> Option<Program<'a>> {
    let signed = handwired_rs_sign::verify(program, signature, &TRUSTED).ok()?;
    if signed.slot != "heater" || signed.sequence < last_sequence {
        return None;
    }
    // Record signed.sequence as the slot's last, and log the program's hash, before running it.
    Program::parse(program).ok()
}
```

A device checks each program once, when it is installed or loaded, not on every run. Next to the runner, checking
a signature adds 15 to 19 KiB of code and 2 KiB of constants on a Cortex-M4 or M33 (opt-level 3, `s` or `z`; the
crate builds `ed25519-compact` with its `opt_size` feature, without which opt-level 3 adds up to 360 KiB), and takes
about 85 µs for a 12 KiB program on a recent x86 laptop. A device with secure boot may instead check signatures with
the code and keys it already trusts, since the format is plain Ed25519.

## Format

```text
"HWCSIG\0\0" ‖ version (u32 LE, 1) ‖ signer's public key (32) ‖ sequence (u64 LE) ‖ slot length (1) ‖ slot (≤ 64,
UTF-8) ‖ signature (64)
```

The public key in the file only says which key signed: [`verify`] accepts the signature only under a key in its
trusted list. `tests/independent_check.py` rebuilds a pinned signature file from this description with another
Ed25519 implementation (Python's `cryptography`) and checks it byte for byte.

[`verify`]: https://docs.rs/handwired-rs-sign/latest/handwired_rs_sign/fn.verify.html
