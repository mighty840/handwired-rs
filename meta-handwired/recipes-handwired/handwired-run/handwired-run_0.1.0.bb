SUMMARY = "Run circuits exported by handwired-rs"
DESCRIPTION = "handwired-run runs a program exported by handwired-rs (a .hwc file) over records of input cells read \
from standard input, one per line, and prints the output cells. Results are bit-exact with handwired-rs's CPU \
executor; inputs outside the ranges the program declares, and damaged files, are refused."
HOMEPAGE = "https://github.com/mighty840/handwired-rs"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://LICENSE;md5=1a9a7e7ebdb4a06aacd3f5cda2d45c62"

SRC_URI = "crate://crates.io/handwired-rs-run/${PV};name=handwired-rs-run"
SRC_URI[handwired-rs-run.sha256sum] = "5c062c6a0f8892021de8eea35851cb18fd6c0a5f19a63af6ea50056fa8c58b80"
S = "${CARGO_VENDORING_DIRECTORY}/handwired-rs-run-${PV}"

inherit cargo

# The command needs the crate's std feature; the crate has no dependencies.
CARGO_BUILD_FLAGS:append = " --features std --bin handwired-run"

do_install() {
    install -D -m 0755 ${B}/target/${CARGO_TARGET_SUBDIR}/handwired-run ${D}${bindir}/handwired-run
}

BBCLASSEXTEND = "native nativesdk"
