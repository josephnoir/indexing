/******************************************************************************
 * function: compute_column_length                                            *
 * input:                                                                     *
 * output:                                                                    *
 ******************************************************************************/


/**
 * High level function:
 * size_t compute_column_length(global uint* input, global uint* chids,
 *                             global uint* offsets, size_t li, size_t work_size);
 *
 */

/**
 * Steps
 * - create_tmp_array
 * - lazy_segmented_scan (SUM)
 * - stream compaction to reduce all that have a 0 in heads?
 * - exclusive scan on the tmp array
 * (There is probably a more efficient solution for segmented scan)
 */

kernel void column_prepare(global const uint* restrict chids,
                           global const uint* restrict input,
                           global       uint* restrict tmp,
                           global       uint* restrict heads,
                           uint len) {
  const uint n_half = get_global_size(0);
  const uint ai = get_global_id(0);
  const uint bi = get_global_id(0) + n_half;
  if (ai < len) {
    tmp[ai] = 1 + (chids[ai] != 0);
    heads[ai] = (ai == 0) || (input[ai] != input[ai - 1]);
  }
  if (bi < len) {
    tmp[bi] = 1 + (chids[bi] != 0);
    heads[bi] = (input[bi]!= input[bi - 1]);
  }
/*
  uint idx = get_global_id(0);
  tmp[idx] = 1 + (chids[idx] != 0);
  // create heads array for next steps
  heads[idx] = (idx == 0) || (input[idx] != input[idx - 1]);
*/
}

// Replacing the lazy segmented scan above with the segemented scan in the related cl
// file, leaves us with an array that saves the max at the end of each segment. This
// kernel converts the heads array appropriately. Better solution: implement a proper
// segmented reduction.
kernel void convert_heads(global const uint* restrict in,
                          global       uint* restrict out,
                          const uint len) {
  const uint thread = get_global_id(0);
  const uint even = 2 * thread;
  const uint odd  = 2 * thread + 1;
  if (even < len) out[even] = (even == len - 1) ? 1 : in[even + 1];
  if (odd  < len) out[odd ] = (odd  == len - 1) ? 1 : in[odd  + 1];
}

/*
 * Use the stream compaction in the related .cl files
 */

// TODO: exclusive scan algorithm

