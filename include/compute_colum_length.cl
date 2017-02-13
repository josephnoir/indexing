/******************************************************************************
 * function: compute_colum_length                                             *
 * input:                                                                     *
 * output:                                                                    *
 ******************************************************************************/


/**
 * High level function:
 * size_t compute_colum_length(global uint* input, global uint* chids,
 *                             global uint* offsets, size_t li, size_t work_size);
 *
 */


/*
size_t compute_colum_length(global uint* input, global uint* chids,
                            global uint* offsets, size_t li, size_t work_size) {
  local uint tmp[WORK_GROUP_SIZE];
  tmp[li] = (1 + (chids[li] != 0)); // ? 0 : 1));
  barrier(FENCE_TYPE);
  uint keycnt = reduce_by_key_SUM(input, tmp, li, work_size);
  // inclusive scan to create offsets, should be parallel
  offsets[li] = 0;
  barrier(FENCE_TYPE);
  if (li == 0) {
    for (uint i = 1; i < keycnt; ++i) {
      offsets[i] = offsets[i - 1] + tmp[i - 1];
    }
  }
  barrier(FENCE_TYPE);
  return keycnt;
}
*/
