SUMMARY = "Sign handwired programs and check their signatures"
DESCRIPTION = "handwired-sign checks that a program exported by handwired-rs (a .hwc file) was signed by a trusted \
key, for its slot, and prints the slot and release number it was signed for; it also makes keys and signs programs. \
Signatures are Ed25519, in a file beside each program (heater.hwc.sig)."
HOMEPAGE = "https://github.com/mighty840/handwired-rs"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://LICENSE;md5=1a9a7e7ebdb4a06aacd3f5cda2d45c62"

SRC_URI = "crate://crates.io/handwired-rs-sign/${PV};name=handwired-rs-sign \
           crate://crates.io/ed25519-compact/2.6.0 \
"
# The published crate's checksum, filled in once 0.1.0 is on crates.io. Until then CI builds this recipe from the
# crate the repository packages, with that crate's checksum.
SRC_URI[handwired-rs-sign.sha256sum] = "0000000000000000000000000000000000000000000000000000000000000000"
SRC_URI[ed25519-compact-2.6.0.sha256sum] = "1454db4f2edb7f0e8fe0c5b375b0c978fc63244cc9f010d160e417eb10139aa8"
S = "${CARGO_VENDORING_DIRECTORY}/handwired-rs-sign-${PV}"

inherit cargo

# The command needs the crate's std feature; its one dependency, ed25519-compact, has none of its own.
CARGO_BUILD_FLAGS:append = " --features std --bin handwired-sign"

do_install() {
    install -D -m 0755 ${B}/target/${CARGO_TARGET_SUBDIR}/handwired-sign ${D}${bindir}/handwired-sign
}

BBCLASSEXTEND = "native nativesdk"
