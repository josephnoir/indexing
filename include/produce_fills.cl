/******************************************************************************
 * Copyright (C) 2017                                                         *
 * Raphael Hiesgen <raphael.hiesgen (at) haw-hamburg.de>                      *
 *                                                                            *
 * Distributed under the terms and conditions of the BSD 3-Clause License.    *
 *                                                                            *
 * If you did not receive a copy of the license files, see                    *
 * http://opensource.org/licenses/BSD-3-Clause and                            *
 ******************************************************************************/

/******************************************************************************
 * function: produce_fills                                                    *
 * input:                                                                     *
 * output:                                                                    *
 ******************************************************************************/


/**
 * High level function:
 * void produce_fills(global uint* input, global uint* chids,
 *                    size_t li, size_t work_size);
 */

kernel void produce_fills(global const uint* restrict input,
                          global const uint* restrict chids,
                          global       uint* restrict output,
                          private uint k) {
  const uint thread = get_global_id(0);
  const uint idx_a = 2 * thread;
  const uint idx_b = 2 * thread + 1;
  if (idx_a < k) {
    output[idx_a] = (idx_a != 0 && input[idx_a] == input[idx_a - 1])
                  ? chids[idx_a] - chids[idx_a - 1] - 1
                  : chids[idx_a];
  }
  if (idx_b < k) {
    output[idx_b] = (input[idx_b] == input[idx_b - 1])
                  ? chids[idx_b] - chids[idx_b - 1] - 1
                  : chids[idx_b];
  }
  /*
  uint idx = get_global_id(0);
  if (idx < k) {
    output[idx] = (idx != 0 && input[idx] == input[idx - 1])
                ? chids[idx] - chids[idx - 1] - 1
                : chids[idx];
  }
  */
  /*
  if (idx < k) {
    if (idx != 0 && input[idx] == input[idx - 1])
      output[idx] = chids[idx] - chids[idx - 1] - 1;
    else
      output[idx] = chids[idx];
  }
  */
}

