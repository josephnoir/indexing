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
  uint thread = get_global_id(0);
  if (thread < k) {
    index[2 * thread    ] = chids[thread];
    index[2 * thread + 1] = lits[thread];
  }
}

/**
 * Use stream compaction kernel for the rest of this part
 */

