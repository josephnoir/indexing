/******************************************************************************
 * function: produce_chunk_id_literals                                       *
 * input:                                                                     *
 * output:                                                                    *
 ******************************************************************************/

kernel void produce_chunks(global uint* rids, global uint* chids,
                           global uint* lits) {
  uint idx = get_global_id(0);
  lits[idx] = 1u << (rids[idx] % 31u);
  lits[idx] |= 1u << 31;
  chids[idx] = (uint) rids[idx] / 31;
}

