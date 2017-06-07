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
 * function: radix sort                                                       *
 * input:                                                                     *
 * output:                                                                    *
 ******************************************************************************/

 // Adapted from: https://github.com/Schw1ng/GPURadixSort
 // Also, see:
 // - http://http.developer.nvidia.com/GPUGems3/gpugems3_ch32.html
 // - http://www.heterogeneouscompute.org/wordpress/wp-content/uploads/2011/06/RadixSort.pdf
 // TODO: Can we replace some barriers with mem_fences?

typedef struct radix_config {
  uint radices;  // number of radices
  uint blocks;   // number of blocks
  uint gpb;      // groups per block
  uint tpg;      // threads per group
  uint epg;      // elements per group
  uint rpb;      // radices per block
  uint mask;     // bit mask
  uint l_val;    // L
  uint tpb;      // threads per block =?= get_local_size(0)
  uint size;     // total elements
} configuration;

inline void __attribute__((always_inline))
prefix_sum(local uint* data, int len, int threads) {
  uint thread = get_local_id(0);
  int inc = 2;
  // reduce
  while (inc <= len) {
    int j = inc >> 1;
    for (int i = (j - 1) + (thread * inc); (i + inc) < len; i += (threads * inc))
      data[i + j] = data[i] + data[i + j];
    inc = inc << 1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  // downsweep
  data[len - 1] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  while (inc >= 2) {
    int j = inc >> 1;
    for (int i = (j - 1) + (thread * inc); (i + j) <= len; i += (threads * inc)) {
      uint tmp = data[i + j];
      data[i + j] = data[i] + data[i + j];
      data[i] = tmp;
    }
    inc = inc >> 1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

// Optimization: count in local memory and copy to global when the
// block is finished.
kernel void count(global uint* restrict cell_in,
                  global uint* restrict counters,
                  local  uint*          histogram,
                  configuration conf, uint offset) {
  const uint thread = get_local_id(0);
  const uint block = get_group_id(0);
  const uint group = thread / conf.tpg;
  // Set counters to zero
  for (uint r = 0; r < conf.radices; ++r)
    histogram[conf.radices * group + r] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  // Each group has its own counter for each radix.
  // Position in counter for the first group in this block.
  const uint group_offset = block * conf.gpb;
  // Thread id related to group (not block).
  const uint group_thread = thread % conf.tpg;
  // Start of the elements inside the block
  const uint elem_offset = (group_offset + group) * conf.epg;
  // Number of counters for each radix.
  const uint entries = conf.blocks * conf.gpb;
  // Startindex for the current thread.
  const uint start = elem_offset + group_thread;
  const uint end = min(elem_offset + conf.epg, conf.size);
  for (uint i = start; i < end; i += conf.tpg) {
    const uint bits = (cell_in[i] >> offset) & conf.mask;
    const uint index = (conf.radices * group + bits);
    // The following code ensures that the counters of each thread group are
    // sequentially incremented. Only one thread per group writes in each loop.
    for (uint j = 0; j < conf.tpg; ++j) {
      if (group_thread == j)
        ++histogram[index];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  for (uint r = 0; r < conf.radices; ++r) {
    const uint from = conf.radices * group + r;
    const uint to = r * entries + group_offset + group;
    // const uint index = (bits * entries) + group_offset + group
    counters[to] = histogram[from];
  }
}

kernel void scan(global uint* restrict cell_in,
                 global uint* restrict counters,
                 global uint* restrict prefixes,
                 local uint* l_counters,
                 configuration conf, uint offset) {
  const uint thread = get_local_id(0);
  const uint block = get_group_id(0);
  const uint entries = conf.blocks * conf.gpb; // one entry per group
  const uint radix_base = conf.rpb * block; // radix we handle here
  for (uint i = 0; i < conf.rpb; ++i) {
    const uint radix = radix_base + i;
    const uint start = (radix * entries) + thread;
    const uint end = (radix + 1) * entries;
    const uint prior = counters[((radix + 1) * entries) - 1];
    // Copy entries counters of the radix to local memory
    int k = 0;
    for (uint j = start; j < end; j+= conf.tpb) {
      l_counters[conf.tpb * k + thread] = counters[j];
      ++k;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    prefix_sum(l_counters, entries, conf.tpb);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thread == 1)
      prefixes[radix] = l_counters[entries - 1] + prior;
    barrier(CLK_GLOBAL_MEM_FENCE);
    k = 0;
    for (uint j = start; j < end; j += conf.tpb) {
      counters[j] = l_counters[thread + conf.tpb * k];
      ++k;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
  }
}


kernel void reorder(global uint* restrict cell_in,
                    global uint* restrict cell_out,
                    global uint* restrict counters,
                    global uint* restrict prefixes,
                    local uint* l_counters, local uint* l_prefixes,
                    configuration conf, uint offset) {
  const uint thread = get_local_id(0);
  const uint block = get_group_id(0);
  const uint group = thread / conf.tpg;
  const uint radix = conf.rpb * block;
  const uint entries = conf.gpb * conf.blocks;
  // Finish phase 2 by combining the sum of each radix
  for (uint i = thread; i < conf.radices; i += conf.tpb) {
    l_prefixes[i] = prefixes[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  prefix_sum(l_prefixes, conf.radices, conf.tpb);
  barrier(CLK_LOCAL_MEM_FENCE);
  // Load (groups per block * radices) counters, i.e., the block column
  for (uint i = 0; i < conf.gpb; ++i) {
    if (thread < conf.radices) {
      l_counters[thread * conf.gpb + i]
        = counters[thread * entries + block * conf.gpb + i] + l_prefixes[thread];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint group_offset = block * conf.gpb;
  const uint group_thread = thread % conf.tpg;
  const uint elem_offset = (group_offset + group) * conf.epg;
  const uint start = elem_offset + group_thread;
  const uint end = min(elem_offset + conf.epg, conf.size);
  for (uint i = start; i < end; i += conf.tpg) {
    const uint bits = (cell_in[i] >> offset) & conf.mask;
    const uint index = (bits * conf.gpb) + group;
    for (uint j = 0; j < conf.tpg; ++j) {
      if (group_thread == j) {
        cell_out[l_counters[index]] = cell_in[i];
        ++l_counters[index];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
}

kernel void reorder_kv(global uint* restrict cell_in,
                       global uint* restrict cell_out,
                       global uint* restrict value_in,
                       global uint* restrict value_out,
                       global uint* restrict counters,
                       global uint* restrict prefixes,
                       local uint* l_counters,
                       local uint* l_prefixes,
                       configuration conf, uint offset) {
  const uint thread = get_local_id(0);
  const uint block = get_group_id(0);
  const uint group = thread / conf.tpg;
  const uint radix = conf.rpb * block;
  const uint entries = conf.gpb * conf.blocks;
  // Finish phase 2 by combining the sum of each radix
  for (uint i = thread; i < conf.radices; i += conf.tpb) {
    l_prefixes[i] = prefixes[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  prefix_sum(l_prefixes, conf.radices, conf.tpb);
  barrier(CLK_LOCAL_MEM_FENCE);
  // Load (groups per block * radices) counters, i.e., the block column
  for (uint i = 0; i < conf.gpb; ++i) {
    if (thread < conf.radices) {
      l_counters[thread * conf.gpb + i]
        = counters[thread * entries + block * conf.gpb + i] + l_prefixes[thread];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint group_offset = block * conf.gpb;
  const uint group_thread = thread % conf.tpg;
  const uint elem_offset = (group_offset + group) * conf.epg;
  const uint start = elem_offset + group_thread;
  const uint end = min(elem_offset + conf.epg, conf.size);
  for (uint i = start; i < end; i += conf.tpg) {
    const uint bits = (cell_in[i] >> offset) & conf.mask;
    const uint index = (bits * conf.gpb) + group;
    for (uint j = 0; j < conf.tpg; ++j) {
      if (group_thread == j) {
        cell_out[l_counters[index]] = cell_in[i];
        value_out[l_counters[index]] = value_in[i];
        ++l_counters[index];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
}

