SUMMARY = "Example handwired program: a heater's duty from a temperature and a setpoint"
DESCRIPTION = "heater.hwc is exported by handwired-rs from an ordinary Rust function \
(run/examples/export_heater.rs in handwired-rs), checked against it on 100,000 inputs first. \
Its inputs are the temperature in tenths of a degree (-400..=1250), the setpoint in degrees (5..=30) \
and a boost flag (0 or 1); its outputs are the duty (0..=100) and whether to cool (0 or 1)."
HOMEPAGE = "https://github.com/mighty840/handwired-rs"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

# Signed for slot "heater", release 1, by example.pub. The example's secret key was discarded: sign your own programs
# with a key of your own (handwired-sign keygen) and replace both files.
SRC_URI = "file://heater.hwc file://heater.hwc.sig file://example.pub"
HANDWIRED_PROGRAMS = "heater.hwc"
HANDWIRED_TRUSTED_KEYS = "example.pub"

inherit handwired-program
