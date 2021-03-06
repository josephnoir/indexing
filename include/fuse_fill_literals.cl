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
                          global uint2* restrict index,
                          private uint k) {
  uint thread = get_global_id(0);
  uint2 res;
  if (thread < k) {
    res.x = chids[thread];
    res.y = lits[thread];
    index[thread] = res;
  }
}

/**
 * Use stream compaction kernel for the rest of this part
 */

