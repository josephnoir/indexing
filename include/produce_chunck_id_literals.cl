/******************************************************************************
 * function: produce_chunck_id_literals                                       *
 * input:                                                                     *
 * output:                                                                    *
 ******************************************************************************/

kernel void produce_chuncks(global uint* rids, global uint* chids,
                            global uint* lits) {
  uint index = get_global_id(0);
  lits[index] = 1u << (rids[index] % 31u);
  lits[index] |= 1u << 31;
  chids[index] = (uint) rids[index] / 31;
}
