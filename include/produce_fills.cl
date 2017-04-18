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

kernel void produce_fills(global uint* restrict input,
                          global uint* restrict chids,
                          global uint* restrict output,
                          private uint k) {
  uint thread = get_global_id(0);
  uint idx_a = 2 * thread;
  uint idx_b = 2 * thread + 1;
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

