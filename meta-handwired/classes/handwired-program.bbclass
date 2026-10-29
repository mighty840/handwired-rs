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
#
# Proof reports: a program's report (PROGRAM.proof.json, the format in handwired-rs-sign's README), also in SRC_URI and
# named in HANDWIRED_REPORTS, is checked at build time to name the program's SHA-256, installed beside it and listed
# in the manifest. The image's SPDX documents then list the report with its hash like any other file, and the
# recipe's SPDX document carries the annotation HANDWIRED_REPORTS=... that pairs each report with its program.

HANDWIRED_PROGRAMS ?= ""
HANDWIRED_DIR ?= "${datadir}/handwired"
HANDWIRED_TRUSTED_KEYS ?= ""
HANDWIRED_KEYS_DIR ?= "${sysconfdir}/handwired/trusted"
HANDWIRED_REPORTS ?= ""
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
    programs = d.getVar('HANDWIRED_PROGRAMS').split()
    for report in d.getVar('HANDWIRED_REPORTS').split():
        if not report.endswith('.proof.json') or report[:-len('.proof.json')] not in programs:
            bb.fatal("%s: %s in HANDWIRED_REPORTS isn't PROGRAM.proof.json for a program in HANDWIRED_PROGRAMS"
                     % (d.getVar('PN'), report))
    if d.getVar('HANDWIRED_REPORTS').split():
        d.appendVar('SPDX_CUSTOM_ANNOTATION_VARS', ' HANDWIRED_REPORTS')
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
    for report in ${HANDWIRED_REPORTS}; do
        program="${report%.proof.json}"
        sum=$(sha256sum "${HANDWIRED_SOURCE_DIR}/$program" | cut -d' ' -f1)
        grep -q "\"sha256\": \"$sum\"" "${HANDWIRED_SOURCE_DIR}/$report" \
            || bbfatal "$report isn't a report on this $program (its SHA-256 is $sum)"
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
    for report in ${HANDWIRED_REPORTS}; do
        install -m 0644 "${HANDWIRED_SOURCE_DIR}/$report" ${D}${HANDWIRED_DIR}/
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
    for report in ${HANDWIRED_REPORTS}; do
        sha256sum "$(basename $report)"
    done >> MANIFEST
}

FILES:${PN} += "${HANDWIRED_DIR} ${HANDWIRED_KEYS_DIR}"

