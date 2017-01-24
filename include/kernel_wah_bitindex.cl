/*
 * function: wah_bitmap
 * input:    32-bit integer
 * output:   wah compressed bitmap index, offsets for its values
 * table:    
 */

void sort_rids_by_value(__global uint* input, __global  uint* rids,
                        size_t idx, size_t total);
void parallel_selection_sort(__global uint* key, __global uint* data,
                             size_t idx, size_t total);
void produce_chunck_id_literals(__global uint* rids, __global uint* chids,
                                __global uint* lits, size_t idx, size_t total);

__kernel void kernel_wah_index(__global uint* input,
                               __global uint* rids,
                               __global uint* chids,
                               __global uint* lits,
                               __global uint* index,
                               __global uint* offsets,
                               __global uint* config) {
  // just 1 dimension here: 0
  size_t total = get_global_size(0);
  size_t idx = get_global_id(0);
  // read config data
  // assumed to be == total, except idx, which is == 2 x total
  //uint input_size   = config[0];
  //uint rids_size    = config[1];
  //uint chids_size   = config[2];
  //uint lits_size    = config[3];
  //uint index_size   = config[4];
  //uint offsets_size = config[5];

  //__gloabl uint[index_size] rids;
  sort_rids_by_value(input, rids, idx, total);
  barrier(CLK_GLOBAL_MEM_FENCE);
  //produce_chunck_id_literals(rids, chids, lits, idx, total);
  //barrier(CLK_GLOBAL_MEM_FENCE);
/*
  auto k = merged_lit_by_val_chids(input, chids, lits);
  produce_fills(input, chids, k);
  vector<uint> index(2 * k);
  auto idx_length = fuse_fill_literals(chids, lits, index, k);
  vector<uint> offsets(k);
  auto keycnt = compute_colum_length(input, chids, offsets, k);
*/
}

void sort_rids_by_value(__global uint* input, __global uint* rids,
                        size_t idx, size_t total) {
  rids[idx] = idx;
  // sort by input value
  parallel_selection_sort(input, rids, idx, total);
}

void produce_chunck_id_literals(__global uint* rids, __global uint* chids,
                                __global uint* lits, size_t idx, size_t total) {
  (void) total;
  lits[idx] = rids[idx] % 31u;
  lits[idx] |= 1u << 31;
  chids[idx] = (uint) rids[idx] / 31;
}

/**
 * Sort found on: http://www.bealto.com/gpu-sorting_intro.html
 * License:set syntax=:
 *  This code is released under the following license (BSD-style).
 *  --
 *
 *  Copyright (c) 2011, Eric Bainville
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of Eric Bainville nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY ERIC BAINVILLE ''AS IS'' AND ANY
 *  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL ERIC BAINVILLE BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **/

// One thread per record
void parallel_selection_sort(__global uint* key, __global uint* data,
                             size_t idx, size_t total) {
  uint key_value = key[idx];
  uint data_value = data[idx];
  // Compute position of in[i] in output
  int pos = 0;
  for (size_t j = 0; j < total; ++j) {
    uint curr = key[j]; // broadcasted
    bool smaller = (curr < key_value) || (curr == key_value && j < idx);
    pos += (smaller) ? 1 : 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  key[pos] = key_value;
  data[pos] = data_value;
}
