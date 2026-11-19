#!/bin/sh
# Builds Eclipse 4diac FORTE with the handwired module and runs test/handwired_run.fboot, a 4diac application that
# loads heater.hwc into HANDWIRED_RUN, runs it and checks its outputs and a refusal:
#
#     plc/4diac/build.sh WORKDIR
#
# WORKDIR holds FORTE's source (cloned at FORTE_TAG, 3.3.0 by default), its build and the test's run directory.
# JOBS sets the build's parallelism (2 by default). Exits 0 only if every check in the application passed.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
work=${1:?usage: plc/4diac/build.sh WORKDIR}
tag=${FORTE_TAG:-3.3.0}
jobs=${JOBS:-2}

mkdir -p "$work"
work=$(cd "$work" && pwd)
forte="$work/forte-$tag"
if [ ! -d "$forte" ]; then
    git clone --quiet --depth 1 --branch "$tag" https://github.com/eclipse-4diac/4diac-forte.git "$forte"
fi
# The module goes in like FORTE's own: a directory under modules/ and a line in modules/CMakeLists.txt.
rm -rf "$forte/modules/handwired"
cp -R "$here/handwired" "$forte/modules/handwired"
grep -q '^add_subdirectory(handwired)$' "$forte/modules/CMakeLists.txt" ||
    echo 'add_subdirectory(handwired)' >>"$forte/modules/CMakeLists.txt"

(cd "$repo" && cargo build --release -p handwired-rs-run-c)

# FORTE's posix-release preset, which brings the utility, conversion and IEC 61131-3 blocks the test uses, logging
# at INFO, where TEST_CONDITION reports a check that passed.
(cd "$forte" && cmake --preset posix-release -DFORTE_MODULE_HANDWIRED=ON -DHANDWIRED_RS_DIR="$repo" \
    -DFORTE_LOGLEVEL=LOGINFO >/dev/null)
cmake --build "$forte/build/posix-release" --parallel "$jobs" --target forte
binary=$(find "$forte/build/posix-release" -type f -name forte -perm -u+x | head -n 1)

run="$work/run"
rm -rf "$run"
mkdir -p "$run"
cp "$repo/meta-handwired/recipes-handwired/handwired-heater/files/heater.hwc" "$run/"
# The application's last block stops FORTE; a chain that breaks leaves it running until the timeout.
(cd "$run" && timeout 60 "$binary" -f "$here/test/handwired_run.fboot" >"$run/forte.log" 2>&1) || true

passed=$(grep -c 'TEST_CONDITION_PASSED' "$run/forte.log" || true)
failed=$(grep -c 'TEST_CONDITION_FAILED' "$run/forte.log" || true)
if [ "$passed" -ne 6 ] || [ "$failed" -ne 0 ]; then
    cat "$run/forte.log"
    echo "HANDWIRED_RUN in FORTE $tag: $passed of 6 checks passed, $failed failed" >&2
    exit 1
fi
echo "HANDWIRED_RUN in FORTE $tag: all 6 checks passed"
