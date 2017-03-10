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
  uint r_val;    // R, seems to be equal to tpg
  uint tpg;      // threads per group
  uint epg;      // elements per group
  uint rpb;      // radices per block
  uint mask;     // bit mask
  uint l_val;    // L
  uint tpb;      // threads per block
  uint size;     // total elements
} configuration;

kernel void zeroes(global uint* counters) {
  counters[get_global_id(0)] = 0;
}

/*
Start index for threads:
(block_idx * groups_per_block + (thread_idx / r_val)) * elements_per_group + (thread_idx % r_val)
then increment by r_val / threads_per_group
*/

// Prefix sum, Can we optimize this, maybe, write a static
// version that assumes local size <= 1024?
inline void prefix_sum(local uint* data, int len, int threads) {
  uint lid = get_local_id(0);
  int inc = 2;
  // reduce
  while (inc <= len) {
    int j = inc >> 1;
    for (int i = (j - 1) + (lid * inc); (i + inc) < len; i += threads * inc)
      data[i + j] = data[i] + data[i + j];
    inc = inc << 1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  // downsweep
  data[len - 1] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  while (inc >= 2) {
    int j = inc >> 1;
    for (int i = (j - 1) + (lid * inc); (i + j) <= len; i+= threads * inc) {
      uint tmp = data[i + j];
      data[i + j] = data[i] + data[i + j];
      data[i] = tmp;
    }
    inc = inc >> 1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

kernel void count(global uint* cell_in, global volatile uint* counters,
                  configuration conf, uint offset) {
  const uint lid = get_local_id(0);
  const uint gid = get_group_id(0);
  // Current thread group is based of the local id.
  const uint thread_grp = lid / conf.tpg; // conf.r_val;
  // Each Threadgroup has its own counter for each radix and is
  // calculating the offset based on current block / group.
  const uint grp_offset = gid * conf.gpb;
  const uint pos_in_grp = lid % conf.tpg;
  // Startindex of the datablock that corresponds to the Threadblock.
  const int active_block = grp_offset * conf.epg ;
  // Offset inside the block for the threadgroup of the current thread.
  const int active_group = thread_grp * conf.epg;
  // Number of counters for each radix.
  const uint entries = conf.blocks * conf.gpb;
  // Startindex for the current thread.
  const uint start = active_block + active_group + pos_in_grp; // conf.r_val;
  const uint end = min(active_block + active_group + conf.epg, conf.size);
  for (uint i = start; i < end; i += conf.tpg) {
    uint bits = (cell_in[i] >> offset) & conf.mask;
    uint index = (bits * entries) + grp_offset + thread_grp;
    // The following code ensures that the counters of each thread group
    // are sequentially incremented.
    for (uint j = 0; j < conf.tpg; ++j) {
      if (pos_in_grp == j)
        ++counters[index];
      barrier(CLK_GLOBAL_MEM_FENCE); // TODO: not sure if this is needed
    }
    // TODO: Are atomics needed or is the fence enough?
    // atomics could replace the complete loop. Needs some measurements?
    // atomic_inc(&counters[tmp]);
  }
}

kernel void scan(global uint* cell_in, global volatile uint* counters,
                 global uint* prefixes, local uint* l_counters,
                 configuration conf, uint offset) {
  const uint lid = get_local_id(0);
  const uint gid = get_group_id(0);
  const uint entries = conf.blocks * conf.gpb; // one entry per group
  const uint radix_base = conf.rpb * gid; // radix we handle here
  for (uint i = 0; i < conf.rpb; ++i) {
    const uint radix = radix_base + i;
    const uint prior = counters[((radix + 1) * entries) - 1];
    const uint start = (radix * entries) + lid;
    const uint end = (radix + 1) * entries;
    // Copy entries counters of the radix to local memory
    int k = 0;
    for (uint j = start; j < end; j+= conf.tpb) {
      l_counters[conf.tpb * k + lid] = counters[j];
      ++k;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    prefix_sum(l_counters, entries, conf.tpb);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 1)
      prefixes[radix] = l_counters[entries - 1] + prior;
    k = 0;
    for (uint j = start; j < end; j += conf.tpb) {
      counters[j] = l_counters[lid + conf.tpb * k];
      ++k;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
  }
}

kernel void reorder(global uint* cell_in, global uint* cell_out,
                    global volatile uint* counters, global uint* prefixes,
                    local uint* l_counters, local uint* l_prefixes,
                    configuration conf, uint offset) {
  const uint lid = get_local_id(0);
  const uint gid = get_group_id(0);
  const uint thread_grp = lid / conf.tpg; // conf.r_val
  const uint radix = conf.rpb * gid;
  const uint entries = conf.gpb * conf.blocks;
  const uint rc_offset = radix * entries;
  // Final step of sum operation
  for (uint i = 0 ; i < conf.rpb ; ++i) {
    for (uint j = lid; j < entries; j += conf.tpb) {
      l_counters[i * entries + j] = counters[i * entries + rc_offset + j];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // Read radix prefixes to localMemory
  for (uint i = lid; i < conf.radices; i += conf.tpb)
    l_prefixes[i] = prefixes[i];
  // prefix sum on the prefixes in local memory
  barrier(CLK_LOCAL_MEM_FENCE);
  prefix_sum(l_prefixes, conf.radices, conf.tpb);
  barrier(CLK_LOCAL_MEM_FENCE);
  // add prefix sum to the subcounters
  for (uint i = 0; i < conf.rpb; ++i) {
    for (uint j = lid; j < entries; j += conf.tpb) {
      l_counters[i * entries + j] += l_prefixes[radix + i];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // Write back radices at the calculated offsets
  for (uint i = 0; i < conf.rpb; ++i) {
    for (uint j = lid; j < entries; j += conf.tpb) {
      counters[i * entries + rc_offset + j] = l_counters[i * entries + j];
      barrier(CLK_GLOBAL_MEM_FENCE);
    }
  }
  const uint active_counter = gid * conf.gpb;
  const uint active_block = active_counter * conf.epg;
  const uint active_group = thread_grp * conf.epg;
  const uint pos_in_grp = lid % conf.tpg;
  const uint start = active_block + active_group + pos_in_grp; // lid % conf.r_val;
  const uint end = min(active_block + active_group + conf.epg, conf.size);
  for (uint i = start; i < end; i += conf.tpg) {
    const uint bits = (cell_in[i] >> offset) & conf.mask;
    const uint index = bits * entries + active_counter + thread_grp;
    for (uint j = 0; j < conf.tpg; ++j) { // was < conf.r_val
      if (pos_in_grp == j) {
        barrier(CLK_GLOBAL_MEM_FENCE);
        cell_out[counters[index]] = cell_in[i];
        barrier(CLK_GLOBAL_MEM_FENCE);
        atomic_inc(&counters[index]);
        barrier(CLK_GLOBAL_MEM_FENCE);
        //++counters[index];
      }
      barrier(CLK_GLOBAL_MEM_FENCE);
      // barrier(CLK_GLOBAL_MEM_FENCE);
    }
  }
}

kernel void reorder_kv(global uint* cell_in, global uint* cell_out,
                       global uint* value_in, global uint* value_out,
                       global uint* counters, global uint* prefixes,
                       local uint* l_counters, local uint* l_prefixes,
                       configuration conf, uint offset) {
  const uint lid = get_local_id(0);
  const uint gid = get_group_id(0);
  const uint thread_grp = lid / conf.r_val;
  const uint act_radix = conf.rpb * gid;
  const uint entries = conf.gpb * conf.blocks;
  const uint rc_offset = act_radix * entries;
  // Finish the radix sum
  for (uint i = 0 ; i < conf.rpb ; i++) {
    for (uint j = lid; j < entries; j += conf.tpb) {
      l_counters[i * entries + j] = counters[rc_offset + i * entries + j];
    }
  }
  // Read radix prefixes to localMemory
  for (uint i = lid; i < conf.radices; i+= conf.tpb) {
    l_prefixes[i] = prefixes[i];
  }
  // build prefix sum over radix counters?
  // (seems to be the prefixes, not counters)
  barrier(CLK_GLOBAL_MEM_FENCE);
  prefix_sum(l_prefixes, conf.radices, conf.tpb);
  barrier(CLK_GLOBAL_MEM_FENCE);
  // Add the sum of the radix to all subounter of the radixes
  for (uint i = 0; i < conf.rpb; i++) {
    for (uint j = lid; j < entries; j += conf.tpb) {
      l_counters[i * entries + j] += l_prefixes[act_radix + i];
    }
    // barrier(CLK_GLOBAL_MEM_FENCE);
  }
  // Write back the radixes to their respectve offers
  for (uint i = 0; i < conf.rpb; i++) {
    for (uint j = lid; j < entries; j += conf.tpb) {
      // The entries counters of the radix are read from global memory to
      // shared memory.
      counters[rc_offset + i * entries + j] = l_counters[i * entries + j];
    }
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  const uint active_block = gid * conf.gpb * conf.epg ;
  const uint active_counter = gid * conf.gpb  ;
  const uint active_group = (lid / conf.r_val) * conf.epg;
  const uint start = active_block + active_group +  lid % conf.r_val;
  uint end = min(active_block + active_group + conf.epg, conf.size);
  for (uint i = start; i < end; i += conf.tpg) {
    const uint bits = (cell_in[i] >> offset) & conf.mask;
    const uint index = bits * entries + active_counter + thread_grp;
    for (uint j = 0; j < conf.r_val; ++j) {
      if (lid % conf.r_val == j) {
        cell_out[counters[index]] = cell_in[i];
        value_out[counters[index]] = value_in[i];
        //atomic_inc(&counters[index]);
        ++counters[index];
      }
      barrier(CLK_GLOBAL_MEM_FENCE);
    }
  }
}
