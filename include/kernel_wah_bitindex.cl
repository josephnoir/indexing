/******************************************************************************
 * function: wah_bitmap                                                       *
 * input:    32-bit integer                                                   *
 * output:   wah compressed bitmap index, offsets for its values              *
 ******************************************************************************/

#define FENCE_TYPE CLK_LOCAL_MEM_FENCE
#define WORK_GROUP_SIZE 1024

// Processing steps
void sort_rids_by_value(global uint* input, global  uint* rids,
                        size_t li, size_t work_size);
void produce_chunck_id_literals(global uint* rids, global uint* chids,
                                global uint* lits, size_t li, size_t work_size);
size_t merged_lit_by_val_chids(global uint* input, global uint* chids,
                               global uint* lits, size_t li, size_t work_size);
void produce_fills(global uint* input, global uint* chids,
                   size_t li, size_t work_size);
size_t fuse_fill_literals(global uint* chids, global uint* lits,
                          global uint* index, size_t li, size_t work_size);
size_t compute_colum_length(global uint* input, global uint* chids,
                            global uint* offsets, size_t li, size_t work_size);

// helper functions
void parallel_selection_sort(global uint* key, global uint* data,
                             size_t li, size_t work_size);
size_t reduce_by_key_OR(global uint* keys_high, global uint* keys_low,
                        global uint* data, size_t li, size_t work_size);
size_t reduce_by_key_SUM(global uint* keys, local uint* data,
                         size_t li, size_t work_size);

// main kernel
kernel void kernel_wah_index(global uint* config,
                             global uint* input,
                             global uint* index,
                             global uint* offsets,
                             global uint* rids,
                             global uint* chids,
                             global uint* lits) {
  // config structure:
  // config[0] : amount of values to process
  // config[1] : number of work groups
  // config[wg index + 2]     : input length for work group
  // config[wg index + 2 + 1] : index length for work group
  // One-dimensional array here, only dim 0
  // uint num_total = get_global_size(0);
  uint num_local = get_local_size(0);
  // uint gi = get_global_id(0); // global index
  uint li = get_local_id(0);  // local index
  uint wg = get_group_id(0);  // work group index
  uint cfg_pos   = (wg * 2) + 2;
  uint num_wg    = config[1]; // should be <= get_num_groups(0)
  uint work_size = config[cfg_pos];
  uint offset    = wg * num_local;
  // acquire pointers to the memory regions for this work group
  global uint* work_input   = input   + offset;
  global uint* work_index   = index   + (offset * 2);
  global uint* work_offsets = offsets + offset;
  global uint* work_rids    = rids    + offset;
  global uint* work_chids   = chids   + offset;
  global uint* work_lits    = lits    + offset;
  // Process local blocks, 1 work-item per value
  if (li < work_size && wg < num_wg) {
    sort_rids_by_value(work_input, work_rids, li, work_size);
    barrier(FENCE_TYPE);
    produce_chunck_id_literals(work_rids, work_chids, work_lits, li, work_size);
    barrier(FENCE_TYPE);
    size_t k = merged_lit_by_val_chids(work_input, work_chids, work_lits, li,
                                       work_size);
    barrier(FENCE_TYPE);
    produce_fills(work_input, work_chids, li, k);
    barrier(FENCE_TYPE);
    uint length = fuse_fill_literals(work_chids, work_lits, work_index, li, k);
    barrier(FENCE_TYPE);
    uint keycnt = compute_colum_length(work_input, work_chids, work_offsets,
                                       li, k);
    barrier(FENCE_TYPE);
    config[cfg_pos    ] = keycnt;
    config[cfg_pos + 1] = length;
  }
}

void sort_rids_by_value(global uint* input,global uint* rids,
                        size_t li, size_t work_size) {
  rids[li] = li;
  barrier(FENCE_TYPE); // is this needed here?
  // sort by input value
  parallel_selection_sort(input, rids, li, work_size);
}

void produce_chunck_id_literals(global uint* rids, global uint* chids,
                                global uint* lits, size_t li, size_t work_size) {
  (void) work_size;
  lits[li] = 1u << (rids[li] % 31u);
  lits[li] |= 1u << 31;
  chids[li] = (uint) rids[li] / 31;
}

size_t merged_lit_by_val_chids(global uint* input, global uint* chids,
                               global uint* lits, size_t li, size_t work_size) {
  // avoid merge of chids and input into keys,
  // simply pass them both
  size_t k = reduce_by_key_OR(chids, input, lits, li, work_size);
  return k;
}

void produce_fills(global uint* input, global uint* chids,
                   size_t li, size_t work_size) {
  uint tmp = chids[li];
  if (li != 0 && input[li] == input[li - 1]) {
    tmp = chids[li] - chids[li - 1] - 1;
  }
  // This branch leads to loss of fills at the beginning of
  // a bitmap index.
  /* else {
    if (chids[li] != 0) {
      chids[li] = chids[li] - 1;
    }
  }*/
  barrier(FENCE_TYPE);
  chids[li] = tmp;
}

size_t fuse_fill_literals(global uint* chids, global uint* lits,
                          global uint* index, size_t li, size_t work_size) {
  local uint markers [WORK_GROUP_SIZE * 2];
  local uint position[WORK_GROUP_SIZE * 2];
  volatile local int len;
  len = 0;
  uint a =  2 * li;
  uint b = (2 * li) + 1;
  index[a] = chids[li];
  index[b] = lits[li];
  barrier(FENCE_TYPE);
  // stream compaction
  markers[a] = index[a] != 0; // ? 1 : 0;
  markers[b] = index[b] != 0; // ? 1 : 0;
  position[a] = 0;
  position[b] = 0;
  barrier(FENCE_TYPE);
  // should be a parallel scan
  if (li == 0) {
    for (uint i = 1; i < WORK_GROUP_SIZE * 2; ++i) {
      position[i] = position[i - 1] + markers[i - 1];
    }
  }
  // end parallel scan
  uint tmp_a = index[a];
  uint tmp_b = index[b];
  barrier(FENCE_TYPE);
  if (li < work_size) {
    if (markers[a] == 1) {
      index[position[a]] = tmp_a;
      atomic_add(&len, 1);
      //atomic_max(&len, position[a] + 1);
    }
    if (markers[b] == 1) {
      index[position[b]] = tmp_b;
      atomic_add(&len, 1);
      //atomic_max(&len, position[b] + 1);
    }
  }
  // end stream compaction
  barrier(FENCE_TYPE); // <-- should there be a barrier here?
  return len;
}

size_t compute_colum_length(global uint* input, global uint* chids,
                            global uint* offsets, size_t li, size_t work_size) {
  local uint tmp[WORK_GROUP_SIZE];
  tmp[li] = (1 + (chids[li] != 0)); // ? 0 : 1));
  barrier(FENCE_TYPE);
  uint keycnt = reduce_by_key_SUM(input, tmp, li, work_size);
  // inclusive scan to create offsets, should be parallel
  offsets[li] = 0;
  barrier(FENCE_TYPE);
  if (li == 0) {
    for (uint i = 1; i < keycnt; ++i) {
      offsets[i] = offsets[i - 1] + tmp[i - 1];
    }
  }
  barrier(FENCE_TYPE);
  return keycnt;
}

// Helper functions

// we have 64 bit keys consisting of (high << 32 | low), but want to
// avoid the extra copy, so ...
size_t reduce_by_key_OR(global uint* keys_high, global uint* keys_low,
                        global uint* data, size_t li, size_t work_size) {
  local uint heads[WORK_GROUP_SIZE];
  volatile local int k;
  k = 0;
  heads[li] = (li == 0) ||
              (keys_high[li] != keys_high[li - 1]) ||
              (keys_low [li] != keys_low [li - 1]);
  barrier(FENCE_TYPE);
  if (heads[li] != 0) {
    uint val = data[li];
    uint curr = li + 1;
    while (heads[curr] == 0 && curr < work_size) {
      val |= data[curr]; // OR operation
      curr += 1;
    }
    atomic_add(&k, 1);
    data[li] = val;
  }
  barrier(FENCE_TYPE);
  if (li == 0) {
    uint pos = 0;
    for (uint i = 0; i < work_size; ++i) {
      if (heads[i] != 0) {
        keys_high[pos] = keys_high[i];
        keys_low[pos] = keys_low[i];
        data[pos] = data[i];
        pos += 1;
      }
    }
  }
  return (uint) k;
}

size_t reduce_by_key_SUM(global uint* keys, local uint* data,
                         size_t li, size_t work_size) {
  local uint heads[WORK_GROUP_SIZE];
  volatile local int k;
  k = 0;
  //heads[li] = 0;
  //if ((li == 0) || (keys[li] != keys[li - 1]))
  //  heads[li] = 1;
  heads[li] = (li == 0) || (keys[li] != keys[li - 1]);
  barrier(FENCE_TYPE);
  if (heads[li] != 0 && li < work_size) {
    uint val = data[li];
    uint curr = li + 1;
    while (heads[curr] == 0 && curr < work_size) {
      val += data[curr]; // SUM operation
      curr += 1;
    }
    atomic_add(&k, 1);
    data[li] = val;
  }
  barrier(FENCE_TYPE);
  if (li == 0) {
    uint pos = 0;
    for (uint i = 0; i < work_size; ++i) {
      if (heads[i] != 0) {
        keys[pos] = keys[i];
        data[pos] = data[i];
        pos += 1;
      }
    }
  }
  return (uint) k;
}

/**
 * Sort found on: http://www.bealto.com/gpu-sorting_intro.html
 * License:
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

void parallel_selection_sort(global uint* key, global uint* data,
                             size_t li, size_t work_size) {
  uint key_value = key[li];
  uint data_value = data[li];
  // Compute position of in[i] in output
  int pos = 0;
  for (size_t j = 0; j < work_size; ++j) {
    uint curr = key[j]; // broadcasted
    bool smaller = (curr < key_value) || (curr == key_value && j < li);
    pos += (smaller) ? 1 : 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  key[pos] = key_value;
  data[pos] = data_value;
}
