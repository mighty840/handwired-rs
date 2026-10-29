#!/bin/sh
# Runs the Kani proofs of src/proofs.rs one at a time (Kani 0.68.0, CBMC 6.11.0): `./kani.sh [HARNESS...]`, all of
# them by default. Exits 1 if any fails.
#
# The proofs over symbolic files bound each loop by what fits in the file, as CBMC loop labels; a bound too low
# fails the proof's unwinding assertion, so a wrong bound can make a proof fail or slow, never pass wrongly. The
# labels are the mangled names Kani gives this crate's functions: if they change, `goto-instrument --show-loops`
# on the harness's .out file lists the new ones. The largest proof needs about 14 GB of memory. Each log is
# target/kani-HARNESS.log.
set -eu
cd "$(dirname "$0")"

crate=Cs8GJc2iPwn3G_16handwired_rs_run
program=_RNvMs1_${crate}NtB5_7Program
read=_RNvMs0_${crate}NtB5_5Layer4read
check=_RNv${crate}5check
layer_bounds="$read.0:4,$read.1:2,$read.2:2,$check.0:5,$check.1:4,$check.2:8,$check.3:3,$check.4:4,$check.5:2"
layer_bounds="$layer_bounds,$check.6:2,_RNCNv${crate}5checks_0B3_.0:6"
parse_bounds="${program}5parse.0:4,${program}5parse.1:5,${program}5parse.2:15,${program}5parse.3:2,memcmp.0:10"
parse_bounds="$parse_bounds,${program}12input_values.0:5,_RNvNt${crate}6proofs18parse_never_panics.0:15"
# The shaped files are larger than CBMC's default 64 array elements it follows one by one.
shaped="--max-field-sensitivity-array-size 512 --unwindset memcmp.0:10,_RNvMNt${crate}6proofsNtB2_6Writer3any.0:17"

prove() {
    harness=$1
    shift
    start=$(date +%s)
    if cargo kani --lib -Z stubbing --harness "proofs::$harness" --exact "$@" >"$logs/kani-$harness.log" 2>&1; then
        result=SUCCESSFUL
    else
        result=FAILED
        failed=1
    fi
    echo "$harness $result $(($(date +%s) - start)) s"
}

logs=${CARGO_TARGET_DIR:-../target}
mkdir -p "$logs"
if [ $# -eq 0 ]; then
    set -- grid_exponent_finds_exactly_the_allowed_powers_of_two on_grid_is_divisibility \
        f32_at_least_is_the_f64_comparison f32_at_most_is_the_f64_comparison next_f32_is_the_neighbour \
        relu_and_keep_if_are_their_definitions round_boolean_rounds_half_up a_layer_never_panics parse_never_panics \
        every_operation_runs_without_panic wrong_lengths_get_their_errors
fi
failed=0
for harness in "$@"; do
    case $harness in
    a_layer_never_panics) prove "$harness" -Z unstable-options --cbmc-args --unwindset "$layer_bounds" ;;
    parse_never_panics) prove "$harness" -Z unstable-options --cbmc-args --unwindset "$parse_bounds" ;;
    # Without the NaN check: f32 arithmetic gives NaN silently, which isn't a panic. Rust's overflow panics stay.
    every_operation_runs_without_panic) prove "$harness" --no-overflow-checks -Z unstable-options --cbmc-args $shaped ;;
    wrong_lengths_get_their_errors) prove "$harness" -Z unstable-options --cbmc-args $shaped ;;
    *) prove "$harness" ;;
    esac
done
exit $failed
