/*
 * function: wah_bitmap
 * input:    32-bit integer
 * output:   wah compressed bitmap index, offsets for its values
 * table:    
 */

#define FENCE_TYPE CLK_LOCAL_MEM_FENCE
#define WORK_GROUP_SIZE 1024

// Processing steps
void sort_rids_by_value(global uint* input, global  uint* rids,
                        size_t idx, size_t total);
void produce_chunck_id_literals(global uint* rids, global uint* chids,
                                global uint* lits, size_t idx, size_t total);
size_t merged_lit_by_val_chids(global uint* input, global uint* chids,
                               global uint* lits, size_t idx, size_t total);
void produce_fills(global uint* input, global uint* chids,
                   size_t idx, size_t total);
size_t fuse_fill_literals(global uint* chids, global uint* lits,
                          global uint* index, size_t idx, size_t total);
size_t compute_colum_length(global uint* input, global uint* chids,
                            global uint* offsets, size_t idx, size_t total);

// helper functions
void parallel_selection_sort(global uint* key, global uint* data,
                             size_t idx, size_t total);
size_t reduce_by_key_OR(local ulong* keys, global uint* lits,
                        size_t idx, size_t total);
size_t reduce_by_key_SUM(global uint* input, local uint* tmp,
                         size_t idx, size_t total);

// main kernel
kernel void kernel_wah_index(global uint* input,
                             global uint* config,
                             global uint* index,
                             global uint* offsets,
                             global uint* rids,
                             global uint* chids,
                             global uint* lits) {
  // just 1 dimension here: 0
  size_t total = get_global_size(0);
  size_t idx = get_global_id(0);
  if (total > WORK_GROUP_SIZE) {
    offsets[idx] = total;
    index[idx] = WORK_GROUP_SIZE;
    return;
  }
  // CONSIDERATIONS:
  // * some scaling state of the are functions required
  //  - sorting
  //  - scan
  //  - stream compaction
  //  - reduce by key
  // * when do we need barriers?
  //  - at the end of functions, before return?
  //  - can be just assign return values to global stuff
  //  - ...

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
  barrier(FENCE_TYPE);
  //index[idx] = input[idx];
  //offsets[idx] = rids[idx];
  produce_chunck_id_literals(rids, chids, lits, idx, total);
  barrier(FENCE_TYPE);
  //index[idx] = rids[idx];
  //offsets[idx] = lits[idx];
  size_t k = merged_lit_by_val_chids(input, chids, lits, idx, total);
  barrier(FENCE_TYPE);
  /*
  index[idx] = input[idx];
  offsets[idx] = lits[idx];
  config[3] = (uint) k;
  config[5] = (uint) k;
  */
  produce_fills(input, chids, idx, k);
  barrier(FENCE_TYPE);
  config[4] = fuse_fill_literals(chids, lits, index, idx, k);
  uint keycnt = compute_colum_length(input, chids, offsets, idx, k);
  (void) keycnt;
}

void sort_rids_by_value(global uint* input,global uint* rids,
                        size_t idx, size_t total) {
  rids[idx] = idx;
  barrier(FENCE_TYPE); // is this needed here?
  // sort by input value
  parallel_selection_sort(input, rids, idx, total);
}

void produce_chunck_id_literals(global uint* rids, global uint* chids,
                                global uint* lits, size_t idx, size_t total) {
  (void) total;
  lits[idx] = 1u << (rids[idx] % 31u);
  lits[idx] |= 1u << 31;
  chids[idx] = (uint) rids[idx] / 31;
}

size_t merged_lit_by_val_chids(global uint* input, global uint* chids,
                               global uint* lits, size_t idx, size_t total) {
  local ulong keys[WORK_GROUP_SIZE];
  keys[idx] = (((ulong) chids[idx]) << 32) | input[idx];
  barrier(FENCE_TYPE);
  size_t k = reduce_by_key_OR(keys, lits, idx, total);
  barrier(FENCE_TYPE);
  chids[idx] = (uint) (keys[idx] >> 32);
  input[idx] = (uint) keys[idx];
  return k;
}

void produce_fills(global uint* input, global uint* chids,
                   size_t idx, size_t total) {
  /*
  local ulong keys[WORK_GROUP_SIZE];
  keys[idx] = (((ulong) chids[idx]) << 32) | input[idx];
  barrier(FENCE_TYPE);
  local uint heads[WORK_GROUP_SIZE];
  heads[idx] = idx == 0 ? 1 : (keys[idx] - keys[idx - 1]);
  if (heads[idx] == 0) {
  */
  if (idx != 0 && (input[idx] == input[idx - 1] &&
                   chids[idx] == chids[idx - 1])) {
    chids[idx] = chids[idx] - chids[idx - 1] - 1;
  } else {
    if (chids[idx] != 0) {
      chids[idx] = chids[idx] - 1;
    }
  }
}

size_t fuse_fill_literals(global uint* chids, global uint* lits,
                          global uint* index, size_t idx, size_t total) {
  local ulong markers[WORK_GROUP_SIZE * 2];
  local ulong position[WORK_GROUP_SIZE * 2];
  volatile local int len;
  uint a = 2 * idx;
  uint b = 2 * idx + 1;
  // stream compaction
  if (idx < total) {
    index[a] = chids[idx];
    index[b] = lits[idx];
    barrier(FENCE_TYPE);
    markers[a] = index[a] == 0 ? 0 : 1;
    markers[b] = index[b] == 0 ? 0 : 1;
    barrier(FENCE_TYPE);
    // should be a parallel scan
    if (idx == 0) {
      position[0] = 0;
      for (uint i = 1; i < 2 * total; ++i) {
        position[i] = position[i - 1] + markers[i - 1];
      }
    }
    uint tmp_a = index[a];
    uint tmp_b = index[b];
    barrier(FENCE_TYPE);
    if (markers[a] == 1) {
      index[position[a]] = tmp_a;
      atomic_add(&len, 1);
    }
    if (markers[b] == 1) {
      index[position[b]] = tmp_b;
      atomic_add(&len, 1);
    }
  }
  barrier(FENCE_TYPE); // <-- should there be a barrier here?
  return len;
}

size_t compute_colum_length(global uint* input, global uint* chids,
                            global uint* offsets, size_t idx, size_t k) {
  local uint tmp[WORK_GROUP_SIZE];
  tmp[idx] = 1 + (chids[idx] == 0 ? 0 : 1);
  uint keycnt = reduce_by_key_SUM(input, tmp, idx, k);
  // inclusive scan to create offsets
  return keycnt;
}

// Helper implementations

size_t reduce_by_key_OR(local ulong* keys, global uint* lits,
                        size_t idx, size_t total) {
  local uint heads[WORK_GROUP_SIZE];
  heads[idx] = idx == 0 ? 1 : (keys[idx] - keys[idx - 1]);
  volatile local int k;
  if (heads[idx] != 0) {
    uint curr = idx + 1;
    uint val = lits[idx];
    while (heads[curr] == 0 && curr < total) {
      val |= lits[curr];
      curr += 1;
    }
    atomic_add(&k, 1);
    lits[idx] = val;
  }
  barrier(FENCE_TYPE);
  if (idx == 0) {
    uint pos = 0;
    for (uint i = 0; i < total; ++i) {
      if (heads[i] != 0) {
        keys[pos] = keys[i];
        lits[pos] =lits[i];
        pos += 1;
      }
    }
  }
  return (uint) k;
}

size_t reduce_by_key_SUM(global uint* input, local uint* tmp,
                         size_t idx, size_t total) {
  return total;
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
void parallel_selection_sort(global uint* key, global uint* data,
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

