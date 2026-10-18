# Installs programs exported by handwired-rs (.hwc files, from Compiled::export) under ${HANDWIRED_DIR}, with a
# sha256sum manifest, so a device can say which logic it runs (`cd /usr/share/handwired && sha256sum -c MANIFEST`)
# and the image's SPDX output records each program's hash.
#
# A recipe lists the files in SRC_URI and names them in HANDWIRED_PROGRAMS:
#
#   inherit handwired-program
#   SRC_URI = "file://heater.hwc"
#   HANDWIRED_PROGRAMS = "heater.hwc"
#
# Each program is checked at build time with handwired-run-native (checksum, version and structure), so a damaged
# or newer-format file fails the build instead of the device.

HANDWIRED_PROGRAMS ?= ""
HANDWIRED_DIR ?= "${datadir}/handwired"
# Where SRC_URI's files are unpacked: UNPACKDIR from styhead on, WORKDIR before.
HANDWIRED_SOURCE_DIR ?= "${@d.getVar('UNPACKDIR') or d.getVar('WORKDIR')}"
S = "${HANDWIRED_SOURCE_DIR}"

# Programs are data: the same file runs on every architecture.
inherit allarch

DEPENDS += "handwired-run-native"
RDEPENDS:${PN} += "handwired-run"

python () {
    if not d.getVar('HANDWIRED_PROGRAMS').split():
        bb.fatal("%s inherits handwired-program but HANDWIRED_PROGRAMS is empty" % d.getVar('PN'))
}

do_compile() {
    for program in ${HANDWIRED_PROGRAMS}; do
        handwired-run --info "${HANDWIRED_SOURCE_DIR}/$program" > "${B}/$(basename $program).info" \
            || bbfatal "$program isn't a program handwired-run can run"
    done
}

do_install() {
    install -d ${D}${HANDWIRED_DIR}
    for program in ${HANDWIRED_PROGRAMS}; do
        install -m 0644 "${HANDWIRED_SOURCE_DIR}/$program" ${D}${HANDWIRED_DIR}/
    done
    cd ${D}${HANDWIRED_DIR}
    for program in ${HANDWIRED_PROGRAMS}; do
        sha256sum "$(basename $program)"
    done > MANIFEST
}

FILES:${PN} += "${HANDWIRED_DIR}"

