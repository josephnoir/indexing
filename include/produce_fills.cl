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

kernel void produce_fills(global uint* config, global uint* input,
                          global uint* chids,  global uint* output) {
  uint k = config[1];
  uint idx = get_global_id(0);
  if (idx < k) {
    output[idx] = (idx != 0 && input[idx] == input[idx - 1])
                ? chids[idx] - chids[idx - 1] - 1
                : chids[idx];
  }
  /*
  if (idx < k) {
    if (idx != 0 && input[idx] == input[idx - 1])
      output[idx] = chids[idx] - chids[idx - 1] - 1;
    else
      output[idx] = chids[idx];
  }
  */
}

