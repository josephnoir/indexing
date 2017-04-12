/******************************************************************************
 * function: fuse_fill_literals                                               *
 * input:                                                                     *
 * output:                                                                    *
 ******************************************************************************/


/**
 * High level function:
 * size_t fuse_fill_literals(global uint* chids, global uint* lits,
 *                           global uint* index, size_t li, size_t work_size);
 *
 */

kernel void prepare_index(global uint* restrict chids,
                          global uint* restrict lits,
                          global uint* restrict index,
                          private uint k) {
  uint idx = get_global_id(0);
  if (idx < k) {
    index[2 * idx    ] = chids[idx];
    index[2 * idx + 1] = lits[idx];
  }
}

/**
 * Use stream compaction kernel for the rest of this part
 */

