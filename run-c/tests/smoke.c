/* SPDX-License-Identifier: MIT */
/* The C interface from C: `smoke PROGRAM` runs the heater program the Yocto layer installs on a few inputs and
 * checks the outputs and refusals. Built and run by CI (see run-c/README.md). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "handwired_run.h"

static int failures = 0;

static void check(int ok, const char *what) {
  if (!ok) {
    fprintf(stderr, "FAILED: %s\n", what);
    failures++;
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: smoke heater.hwc\n");
    return 2;
  }
  FILE *file = fopen(argv[1], "rb");
  if (!file) {
    perror(argv[1]);
    return 2;
  }
  static uint8_t bytes[1 << 16];
  size_t length = fread(bytes, 1, sizeof bytes, file);
  fclose(file);

  hw_program program;
  hw_error error;
  char text[200];
  check(hw_program_parse(bytes, length, &program, &error) == HW_OK, "parse");
  check(hw_program_inputs(&program) == 3 && hw_program_outputs(&program) == 2, "counts");
  hw_program_description(&program, text, sizeof text);
  check(strncmp(text, "heater", 6) == 0, "description");

  static float scratch[4096];
  size_t scratch_count = hw_program_scratch(&program);
  check(scratch_count > 0 && scratch_count <= 4096, "scratch");

  const float cases[4][5] = {{180, 21, 0, 30, 0}, {180, 21, 1, 50, 0}, {50, 21, 0, 100, 0}, {306, 5, 1, 20, 1}};
  for (int i = 0; i < 4; i++) {
    float outputs[2] = {-1, -1};
    int code = hw_program_run(&program, cases[i], 3, scratch, scratch_count, outputs, 2, &error);
    check(code == HW_OK && outputs[0] == cases[i][3] && outputs[1] == cases[i][4], "outputs");
  }

  float outputs[2] = {-1, -1};
  const float off_grid[3] = {215.5f, 21, 0};
  check(hw_program_run(&program, off_grid, 3, scratch, scratch_count, outputs, 2, &error) == HW_INPUT_OFF_GRID,
        "off-grid refused");
  check(outputs[0] == -1 && outputs[1] == -1, "refused run leaves outputs");
  hw_error_text(&error, text, sizeof text);
  check(strcmp(text, "input cell 0 is 215.5, not a multiple of 1, the step its function declared") == 0,
        "refusal text");

  bytes[100] ^= 1;
  check(hw_program_parse(bytes, length, &program, &error) == HW_CHECKSUM, "changed byte refused");
  check(hw_program_run(&program, cases[0], 3, scratch, scratch_count, outputs, 2, NULL) == HW_NOT_PARSED,
        "unparsed storage refused");

  if (failures == 0) {
    printf("smoke: all checks passed\n");
  }
  return failures == 0 ? 0 : 1;
}
