/* SPDX-License-Identifier: MIT */
/*
 * handwired_run.h: run program files exported by handwired-rs from C or C++.
 *
 * The caller owns every buffer: the program's bytes (which must stay in place while the program is used), its
 * storage, scratch memory, inputs and outputs. Nothing allocates. Parse once, at start-up; run every scan. A run
 * does the same operations in the same order whatever its inputs, and refuses inputs outside the ranges, grids
 * and types its function declared.
 *
 * Link with libhandwired_rs_run_c.a, built by `cargo build --release -p handwired-rs-run-c`.
 */
#ifndef HANDWIRED_RUN_H
#define HANDWIRED_RUN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Storage for a parsed program. Valid after hw_program_parse returns HW_OK. */
typedef struct {
  uint64_t opaque[24];
} hw_program;

/* Why a call failed; hw_error_text describes it. */
typedef struct {
  uint64_t opaque[8];
} hw_error;

/* What calls return. */
enum {
  HW_OK = 0,
  HW_NULL_ARGUMENT = 1,       /* a null pointer where a buffer was needed */
  HW_NOT_PARSED = 2,          /* the storage wasn't filled by hw_program_parse */
  HW_TRUNCATED = 3,           /* the bytes end before the program does */
  HW_NOT_A_PROGRAM = 4,       /* the bytes don't start like a handwired program */
  HW_UNSUPPORTED_VERSION = 5, /* a format version this runner doesn't read */
  HW_CHECKSUM = 6,            /* the bytes changed after the program was exported */
  HW_MALFORMED = 7,           /* the program's parts don't fit together */
  HW_INPUT_LENGTH = 8,        /* not the program's number of input cells */
  HW_INPUT_OUT_OF_RANGE = 9,  /* an input outside the range its function declared */
  HW_INPUT_OFF_GRID = 10,     /* an input between the steps of its grid, e.g. 0.5 for an integer */
  HW_INPUT_OUTSIDE_VARIANT = 11, /* an enum variant's cell set while another variant is selected */
  HW_INPUT_NOT_ALLOWED = 12,  /* an input that isn't one of its declared values */
  HW_SCRATCH_TOO_SMALL = 13,  /* fewer scratch cells than hw_program_scratch */
  HW_OUTPUT_LENGTH = 14,      /* not the program's number of output cells */
  HW_OTHER = 15               /* a reason added after this header */
};

/* Check `length` bytes at `bytes` (checksum, format, every index) and fill `program`. `error` may be NULL. */
int hw_program_parse(const uint8_t *bytes, size_t length, hw_program *program, hw_error *error);

/* The program's input cells, output cells and scratch cells; 0 for storage no parse filled. */
size_t hw_program_inputs(const hw_program *program);
size_t hw_program_outputs(const hw_program *program);
size_t hw_program_scratch(const hw_program *program);

/* Write the program's description (its function's name and signature) to `buffer`, as hw_error_text does. */
size_t hw_program_description(const hw_program *program, char *buffer, size_t size);

/* Run the program. `scratch` holds at least hw_program_scratch(program) cells. Returns HW_OK, or why the run was
 * refused, also written to `error` unless it is NULL; a refused run doesn't write `outputs`. */
int hw_program_run(const hw_program *program, const float *inputs, size_t input_count, float *scratch,
                   size_t scratch_count, float *outputs, size_t output_count, hw_error *error);

/* Describe `error` in `buffer`: the words handwired-run prints, cut to `size` bytes with a terminating NUL.
 * Returns the full length, so a result of `size` or more means it was cut. */
size_t hw_error_text(const hw_error *error, char *buffer, size_t size);

#ifdef __cplusplus
}
#endif

#endif /* HANDWIRED_RUN_H */
