# handwired-rs-sign

Signatures for program files exported by [handwired-rs](https://crates.io/crates/handwired-rs). A device checks
that a program comes from a key it trusts, for the slot it is loaded into, before handing the bytes to
[handwired-rs-run](../run/README.md), and keeps a version log of what it installed. `no_std`, no allocation, no
`unsafe`. Signatures need one dependency, [`ed25519-compact`](https://crates.io/crates/ed25519-compact); the log
(feature `log`) adds [`hmac-sha256`](https://crates.io/crates/hmac-sha256) for SHA-256. Neither has dependencies of
its own.

The runner's checksum catches damaged files, not changed ones: anyone who can write a file can write a valid
checksum. A signature can only be made with the secret key, which stays with whoever releases programs, never on the
device or in the image.

- **Next to the program, not inside it.** `heater.hwc.sig` sits beside `heater.hwc`, whose bytes, and the SHA-256
  that identifies them in manifests and logs, don't change. Runners that don't check signatures read the same files.
- **For one slot.** A signature names the slot the program is for (`estop`, `heater`), so a validly signed program
  can't be loaded in place of another.
- **With a release number.** A signature carries a sequence number. [`verify`] returns it, and the device must refuse
  one below the last it accepted for that slot, or an old, validly signed program could be loaded again. The
  version log below does that.

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

## Version log

With the `log` feature, a device keeps a log of every program presented for each slot, installed or refused, and
`log::admit` decides from it:

- a valid signature under a trusted key, for this slot, is required;
- a release below the slot's installed one is refused, and the installed release number with other bytes too;
  the same program again is a re-install;
- every decision is an entry, refusals included (untrusted key, signature mismatch, other slot, older release,
  release reused, unreadable signature): the evidence of attempted changes.

```sh
handwired-sign log new /var/lib/handwired/log
handwired-sign admit --log /var/lib/handwired/log --key release.pub --slot heater heater.hwc   # exit 0: installed
handwired-sign log show /var/lib/handwired/log
```

An entry is 178 bytes: the device's time, the outcome, the slot, the sequence and key the signature claims, the
program's SHA-256 (the same as in a `sha256sum` manifest) and the hash of the entry before; a header with a 32-byte
log id starts the chain. Ten installs a year for five years is under 9 KiB. The chain shows a changed, removed or
reordered entry, but not entries cut off the end or a whole log swapped for another consistent one: keep the head
(`log check` prints the entry count and last hash) where an attacker can't write it, or report it elsewhere, and
compare. On a microcontroller, `log::admit` works on the log's bytes and returns the entry to append; append it
before acting on the decision.

## Proof reports

A signature says who released a program and the log what a device installed; a proof report says what is known
about what the program computes. It is a JSON file beside the program, `heater.hwc.proof.json`, that names the
program by its SHA-256, so it is about exactly those bytes:

```json
{
  "format": "handwired-proof-report 1",
  "program": { "file": "heater.hwc", "sha256": "8f1076fb…", "bytes": 11914, "format_version": 3 },
  "domain": "every input handwired-rs-run accepts for this file: …",
  "claims": [
    {
      "kind": "exhaustive",
      "statement": "on every input of the domain, the output cells … equal, bit for bit, … what the Rust function returns",
      "method": "enumeration of the domain",
      "tool": "…",
      "inputs": 85852,
      "mismatches": 0,
      "result": "holds"
    }
  ],
  "assumes": ["IEEE 754 binary32 addition and multiplication, round to nearest, subnormals not flushed, …"],
  "not_covered": ["the sign of a zero output", "…"],
  "generated_by": "…"
}
```

`kind` is one of `exhaustive` (every input of the domain run and compared), `equivalence` (proved equal to a
specification, e.g. by an SMT solver), `range-certificate` (every value proved to stay in its range, so no f32
operation rounds), `property` (a statement about one run) and `temporal` (a statement over many runs of a program
with state). A claim proved by a solver names it with its version and adds `"query": { "path": …, "sha256": … }`,
the exact query, so anyone can run it again; `result` is `holds`, `violated` or `unknown`. A report has no dates, so
the same evidence gives the same file.

`cargo run --release -p handwired-rs-sign --features log --example heater_report -- heater.hwc` writes the report of
the Yocto layer's example: it runs every one of the program's 85,852 accepted inputs through handwired-rs-run and
compares the output bits with the Rust function's. The layer checks that each report names its program's hash,
installs it beside the program and lists it in the image's manifest and SPDX documents.

## Format

```text
"HWCSIG\0\0" ‖ version (u32 LE, 1) ‖ signer's public key (32) ‖ sequence (u64 LE) ‖ slot length (1) ‖ slot (≤ 64,
UTF-8) ‖ signature (64)
```

The public key in the file only says which key signed: [`verify`] accepts the signature only under a key in its
trusted list. `tests/independent_check.py` rebuilds a pinned signature file and a pinned version log from their
descriptions with other implementations (Python's `cryptography` and `hashlib`) and checks them byte for byte. The
log's layout is documented in `src/log.rs`.

[`verify`]: https://docs.rs/handwired-rs-sign/latest/handwired_rs_sign/fn.verify.html
