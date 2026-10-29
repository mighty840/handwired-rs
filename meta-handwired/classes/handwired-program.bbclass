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
#
# Signed programs (handwired-rs-sign): name the trusted public keys, also in SRC_URI, in HANDWIRED_TRUSTED_KEYS, and
# every program needs its signature beside it:
#
#   SRC_URI = "file://heater.hwc file://heater.hwc.sig file://release.pub"
#   HANDWIRED_TRUSTED_KEYS = "release.pub"
#
# The build then fails unless each program is signed by one of the keys (handwired-sign-native verify). Signatures
# are installed beside the programs and listed in the manifest, and the keys under HANDWIRED_KEYS_DIR with
# handwired-sign, so the device can check programs it receives later. Sign on the release machine, never in the
# build: the secret key doesn't belong in a layer or a build directory.

HANDWIRED_PROGRAMS ?= ""
HANDWIRED_DIR ?= "${datadir}/handwired"
HANDWIRED_TRUSTED_KEYS ?= ""
HANDWIRED_KEYS_DIR ?= "${sysconfdir}/handwired/trusted"
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
    if d.getVar('HANDWIRED_TRUSTED_KEYS').split():
        d.appendVar('DEPENDS', ' handwired-sign-native')
        d.appendVar('RDEPENDS:' + d.getVar('PN'), ' handwired-sign')
}

do_compile() {
    for program in ${HANDWIRED_PROGRAMS}; do
        handwired-run --info "${HANDWIRED_SOURCE_DIR}/$program" > "${B}/$(basename $program).info" \
            || bbfatal "$program isn't a program handwired-run can run"
        if [ -n "${HANDWIRED_TRUSTED_KEYS}" ]; then
            keys=""
            for key in ${HANDWIRED_TRUSTED_KEYS}; do
                keys="$keys --key ${HANDWIRED_SOURCE_DIR}/$key"
            done
            handwired-sign verify $keys "${HANDWIRED_SOURCE_DIR}/$program" > "${B}/$(basename $program).signed" \
                || bbfatal "$program isn't signed by a key in HANDWIRED_TRUSTED_KEYS"
        fi
    done
}

do_install() {
    install -d ${D}${HANDWIRED_DIR}
    for program in ${HANDWIRED_PROGRAMS}; do
        install -m 0644 "${HANDWIRED_SOURCE_DIR}/$program" ${D}${HANDWIRED_DIR}/
        if [ -n "${HANDWIRED_TRUSTED_KEYS}" ]; then
            install -m 0644 "${HANDWIRED_SOURCE_DIR}/$program.sig" ${D}${HANDWIRED_DIR}/
        fi
    done
    for key in ${HANDWIRED_TRUSTED_KEYS}; do
        install -D -m 0644 "${HANDWIRED_SOURCE_DIR}/$key" ${D}${HANDWIRED_KEYS_DIR}/$(basename $key)
    done
    cd ${D}${HANDWIRED_DIR}
    for program in ${HANDWIRED_PROGRAMS}; do
        sha256sum "$(basename $program)"
        if [ -n "${HANDWIRED_TRUSTED_KEYS}" ]; then
            sha256sum "$(basename $program).sig"
        fi
    done > MANIFEST
}

FILES:${PN} += "${HANDWIRED_DIR} ${HANDWIRED_KEYS_DIR}"

