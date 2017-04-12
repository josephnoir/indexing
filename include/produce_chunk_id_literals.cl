/******************************************************************************
 * function: produce_chunk_id_literals                                       *
 * input:                                                                     *
 * output:                                                                    *
 ******************************************************************************/

kernel void produce_chunks(global uint* restrict rids,
                           global uint* restrict cids,
                           global uint* restrict lits) {
  uint thread = get_global_id(0);
  lits[thread] = 1u << (rids[thread] % 31u);
  lits[thread] |= 1u << 31u;
  cids[thread] = rids[thread] / 31u;
}

