/******************************************************************************
 * function: merge_lit_by_val_chids                                          *
 * input:                                                                     *
 * output:                                                                    *
 ******************************************************************************/

/**
 * High level function:
 * size_t merge_lit_by_val_chids(global uint* input, global uint* chids,
 *                                global uint* lits, size_t li, size_t work_size);
 */


/**
 * Steps
 * - create heads
 * - lazy_segmented_scan (with OR operation)
 * - stream compaction to reduce all that have a 0 in heads?
 * (There is probably a more efficient solution for segmented scan)
 */

kernel void create_heads(global uint* keys_high, global uint* keys_low,
                         global uint* heads) {
  uint idx = get_global_id(0);
  heads[idx] = (idx == 0) ||
               (keys_high[idx] != keys_high[idx - 1]) ||
               (keys_low [idx] != keys_low [idx - 1]);
  /*
  heads[idx] = 0;
  if (idx > 0) {
    if ((keys_high[idx] != keys_high[idx - 1]) ||
        (keys_low [idx] != keys_low [idx - 1]))
    heads[idx] = 1;
  } else {
    heads[idx] = 1;
  }
  */
}

// somewhat inefficient segmented scan (not really a scan as it stores the
// accumulated value at the beginning of the segment)
kernel void lazy_segmented_scan(global uint* heads, global uint* data) {
  uint idx = get_global_id(0);
  uint maximum = get_global_size(0);
  if (heads[idx] != 0) {
    uint val = data[idx];
    uint curr = idx + 1;
    while (heads[curr] == 0 && curr < maximum) {
      val |= data[curr]; // OR operation
      curr += 1;
    }
    data[idx] = val;
  }
}

