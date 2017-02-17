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

/*
void produce_fills(global uint* config, 
                   global uint* input,
                   global uint* chids) {
  uint tmp = chids[li];
  if (li != 0 && input[li] == input[li - 1]) {
    tmp = chids[li] - chids[li - 1] - 1;
  }
  // This branch leads to loss of fills at the beginning of
  // a bitmap index.
  // else {
  //  if (chids[li] != 0) {
  //    chids[li] = chids[li] - 1;
  //  }
  //}
  barrier(FENCE_TYPE);
  chids[li] = tmp;
}
*/
