/******************************************************************************
 * function: compute_colum_length                                             *
 * input:                                                                     *
 * output:                                                                    *
 ******************************************************************************/

/*
 * High level function:
 * kernel void sort_rids_by_value(global uint* config, global uint* input,
 *                                global uint* rids);
 *
 * Tasks:
 * - create vector rids (contains index)
 * - sort input and rids by input
 */

/// Kernels to create rids

kernel void create_rids(gloabl uint* config, global uint* input,
                        gloabl uint* rids) {
  auto max_gi = config[0];
  auto gi = get_global_ids(0);
  if (gi < max_gi) {
    rids[gi] = gi;
  }
}

/// Kernels for sorting


