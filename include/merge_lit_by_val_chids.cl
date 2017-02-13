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


/*
size_t merge_lit_by_val_chids(global uint* input, global uint* chids,
                               global uint* lits, size_t li, size_t work_size) {
  // avoid merge of chids and input into keys,
  // simply pass them both
  size_t k = reduce_by_key_OR(chids, input, lits, li, work_size);
  return k;
}
*/
