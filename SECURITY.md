# Security policy

## Reporting a vulnerability

Report it privately through GitHub: the repository's **Security** tab, then **Report a vulnerability**
(<https://github.com/mighty840/handwired-rs/security/advisories/new>). Please don't open a public issue for it.

Include what you can of:

- which crate or file and which version or commit;
- a program file, signature, log or input that shows it, or the steps that produce one;
- what happens, and what you expected.

We'll confirm we received it, work on a fix with you in the private advisory, and agree on when to publish. Once a
fixed version is out, the advisory is published on GitHub, and for crates on crates.io also reported to the
[RustSec advisory database](https://rustsec.org/). You'll be credited unless you'd rather not be.

## Supported versions

Fixes go into the next release from `main`. Older releases don't get backports.

## What counts

This repository holds the compiler (`handwired-rs`, `handwired-rs-macros`, `handwired-rs-gpu`), the runner that
devices use (`handwired-rs-run`), signatures and the version log (`handwired-rs-sign`) and the Yocto layer
(`meta-handwired`). Vulnerabilities include, for example:

- a program file that makes `handwired-rs-run` panic, read out of bounds, run longer than its structure says, or
  accept an input its checks should refuse;
- a program or signature that `handwired-rs-sign` accepts although it wasn't signed by a trusted key for that slot
  and release, or a log that `admit` or `check` accepts after it was changed;
- something in `meta-handwired` that installs a program the build should have refused.

A circuit that disagrees with its Rust function on an accepted input is a bug in the compiler. Report it as an
ordinary issue, unless you think someone could use it against a device.

Not vulnerabilities: the runner's checksum, which detects damaged files and isn't meant to stop changed ones (that is
what signatures are for), and `meta-handwired/recipes-handwired/handwired-heater/files/example.pub`, whose secret key
was discarded on purpose.
