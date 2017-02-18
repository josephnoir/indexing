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

kernel void prepare_index(global uint* conf, global uint* chids,
                          global uint* lits, global uint* index) {
  uint idx = get_global_id(0);
  uint k = conf[0];
  if (idx < k) {
    index[2 * idx    ] = chids[idx];
    index[2 * idx + 1] = lits[idx];
  } else { // remove this if we shorten the index
    index[2 * idx    ] = 0;
    index[2 * idx + 1] = 0;
  }
}

/**
 * Use stream compaction kernel for the rest of this part
 */

