/******************************************************************************
 * function: produce_chunck_id_literals                                       *
 * input:                                                                     *
 * output:                                                                    *
 ******************************************************************************/

/**
 * High level function:
 * void produce_chunck_id_literals(global uint* rids, global uint* chids,
 *                                 global uint* lits, size_t li, size_t work_size);
 *
 */

/*
void produce_chunck_id_literals(global uint* rids, global uint* chids,
                                global uint* lits, size_t li, size_t work_size) {
  (void) work_size;
  lits[li] = 1u << (rids[li] % 31u);
  lits[li] |= 1u << 31;
  chids[li] = (uint) rids[li] / 31;
}
*/
