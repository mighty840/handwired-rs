# meta-handwired

A Yocto/OpenEmbedded layer for [handwired-rs](https://github.com/mighty840/handwired-rs). With it, business logic
written as an ordinary Rust function, checked against itself and exported as a `.hwc` file, ships in an image as
data:

- `handwired-run`, from the `handwired-rs-run` crate, runs a program over records on standard input. Results are
  bit-exact with handwired-rs's CPU executor. It refuses inputs outside the declared ranges, off their
  grids (e.g. 0.5 for an integer) or that aren't an encoding of their type's values (an enum variant's cells set
  while another variant is selected), and damaged files.
- `handwired-program.bbclass` installs programs under `/usr/share/handwired` with a `sha256sum` manifest. Each file
  is checked with `handwired-run --info` at build time, and the image's SPDX output records each program's hash.
- `handwired-sign`, from the `handwired-rs-sign` crate, checks a program's Ed25519 signature: that a trusted key
  signed it, for which slot and release. With `HANDWIRED_TRUSTED_KEYS`, the class refuses at build time any program
  not signed by one of those keys, and installs the signatures and keys so the device can check later updates.
- `handwired-heater` is an example program, exported by `run/examples/export_heater.rs` and signed.

## Use

Add the layer (it depends only on `core`) and a recipe for your programs:

```bitbake
SUMMARY = "Our thermostat logic"
LICENSE = "CLOSED"
SRC_URI = "file://thermostat.hwc"
HANDWIRED_PROGRAMS = "thermostat.hwc"
inherit handwired-program
```

Export the program on your development machine with `compiled.export("thermostat")` after `compiled.verify(..)`,
and commit the file next to the recipe. On the device:

```sh
echo '215 20 1' | handwired-run /usr/share/handwired/thermostat.hwc
cd /usr/share/handwired && sha256sum -c MANIFEST
```

The checksum and the manifest catch damaged files, not changed ones. To ship only programs you signed, make a key
once and keep its secret half off the build machine, sign each release, and commit the signature and the public key
next to the program:

```sh
handwired-sign keygen release.key release.pub
handwired-sign sign --key release.key --slot thermostat --sequence 3 thermostat.hwc   # thermostat.hwc.sig
```

```bitbake
SRC_URI = "file://thermostat.hwc file://thermostat.hwc.sig file://release.pub"
HANDWIRED_PROGRAMS = "thermostat.hwc"
HANDWIRED_TRUSTED_KEYS = "release.pub"
```

The build fails unless every program is signed by a trusted key. On the device, a program that arrives later is
checked with `handwired-sign verify --key /etc/handwired/trusted/release.pub thermostat.hwc`, which prints its slot
and release; refusing a release lower than the installed one is up to the device's update logic.

## Build and test

CI (`.github/workflows/yocto.yml`) builds `core-image-minimal` for `qemux86-64` on scarthgap with kas, reusing the
Yocto Project's shared state, and runs the layer's runtime tests (`lib/oeqa/runtime/cases/handwired.py`) on it
under QEMU:

```sh
kas build meta-handwired/kas/scarthgap-qemux86-64.yml
kas shell meta-handwired/kas/scarthgap-qemux86-64.yml -c "bitbake core-image-minimal -c testimage"
```

## Status

- Tested only on scarthgap (Rust 1.75): `handwired-rs-run` and `handwired-rs-sign` build with Rust 1.75.
- `handwired-rs-run` and `handwired-rs-sign` aren't on crates.io yet, so their recipes' checksums are placeholders
  until they are. CI builds the recipes from the crates this repository packages: it writes each package's checksum
  into its recipe and puts the package in the download directory, where the `crate://` fetcher finds it.
- The example's signing key was discarded after signing `heater.hwc`; replace `example.pub` and `heater.hwc.sig`
  with your own.
- Exporting programs during the image build, from a native recipe that runs the Rust function, isn't supported yet.
  Programs are exported on a development machine and committed.
