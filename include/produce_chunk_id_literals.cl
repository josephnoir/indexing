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

