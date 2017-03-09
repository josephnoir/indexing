/******************************************************************************
 * function: radix sort                                                       *
 * input:                                                                     *
 * output:                                                                    *
 ******************************************************************************/

 // Adapted from: https://github.com/Schw1ng/GPURadixSort
 // Also, see:
 // - http://http.developer.nvidia.com/GPUGems3/gpugems3_ch32.html
 // - http://www.heterogeneouscompute.org/wordpress/wp-content/uploads/2011/06/RadixSort.pdf

typedef struct radix_config {
  uint radices;  // number of radices
  uint blocks;   // number of blocks
  uint gpb;      // groups per block
  uint r_val;    // R
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

// If this is a prefix sum, can be optimize this, maybe, write a static
// version that assumes local size <= 1024?
inline void prefix_local(local uint* data, int len, int threads) {
  uint lid = get_local_id(0);
  int inc = 2;
  // reduce
  while (inc <= len) {
    int j = inc >> 1;
    for (int i = (j - 1) + (lid * inc); (i + inc) < len; i += threads * inc)
      data[i + j] = data[i] + data[i + j];
    inc = inc << 1;
    barrier(CLK_GLOBAL_MEM_FENCE);
  }
  // downsweep
  data[len - 1] = 0;
  barrier(CLK_GLOBAL_MEM_FENCE);
  while (inc >= 2) {
    int j = inc >> 1;
    for (int i = (j - 1) + (lid * inc); (i + j) <= len; i+= threads * inc) {
      uint tmp = data[i + j];
      data[i + j] = data[i] + data[i + j];
      data[i] = tmp;
    }
    inc = inc >> 1;
    barrier(CLK_GLOBAL_MEM_FENCE);
  }
}

kernel void count(global uint* cell_in,
                  global volatile uint* counters,
                  configuration conf,
                  uint offset) {
  uint lid = get_local_id(0);
  uint gid = get_group_id(0);
  // Current thread group is based of the local id.
  uint thread_grp = lid / conf.r_val;
  // Startindex of the datablock that corresponds to the Threadblock.
  int active_block = gid * conf.gpb * conf.epg ;
  // Offset inside the block for the threadgroup of the current thread.
  int active_group = (lid / conf.r_val) * conf.epg;
  // Number of counters for each radix.
  uint groups = conf.blocks * conf.gpb;
  // Each Threadgroup has its own counter for each radix and is
  // calculating the offset based on current block / group.
  uint grp_offset = gid * conf.gpb;
  // Startindex for the current thread.
  uint start = active_block + active_group + lid % conf.r_val;
  uint end = min(active_block + active_group + conf.epg, conf.size);
  for (;start < end; start += conf.tpg) {
    uint radix = (cell_in[start] >> offset) & conf.mask;
    // The following code ensures that the counters of each Threadgroup
    // are sequentially incremented.
    for (uint i = 0; i < conf.r_val; i++) {
      if (lid % conf.r_val == i) {
        uint tmp = (radix * groups) + grp_offset + thread_grp;
        // TODO: Are atomics needed or is the fence enough?
        //atomic_inc(&counters[tmp]);
        ++counters[tmp];
      }
      barrier(CLK_GLOBAL_MEM_FENCE);
    }
  }
}

kernel void scan(global uint* cell_in,
                 global volatile uint* counters,
                 global uint* prefixes,
                 local uint* l_counters,
                 configuration conf,
                 uint offset) {
  const uint lid = get_local_id(0);
  const uint gid = get_group_id(0);
  const uint groups = conf.blocks * conf.gpb;
  uint radix = conf.rpb * gid;
  for (uint i = 0; i < conf.rpb; ++i) {
    uint start = (radix * groups) + lid;
    uint end = (radix + 1) * groups;
    // Copy groups counters of the radix to local memory
    int k = 0;
    for (uint j = start; j < end; j+= conf.tpb) {
      l_counters[lid + conf.tpb * k] = counters[j];
      ++k;
      // barrier(CLK_LOCAL_MEM_FENCE); // TODO: ?
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // prefix sum on the counters in local memory
    prefix_local(l_counters, groups, conf.tpb);
    barrier(CLK_LOCAL_MEM_FENCE);
    // Gesamtprefix für den aktuellen radix berechnen
    if (lid == 1) {
      prefixes[radix]
        = l_counters[groups - 1] + counters[((radix + 1) * groups) - 1];
    }
    // Write prefix sum back to global memory
    barrier(CLK_GLOBAL_MEM_FENCE);
    k = 0;
    for (uint j = start; j < end; j += conf.tpb) {
      counters[j] = l_counters[lid + conf.tpb * k];
      ++k;
      // barrier(CLK_GLOBAL_MEM_FENCE); // TODO: ?
    }
    ++radix;
    barrier(CLK_GLOBAL_MEM_FENCE);
  }
}

kernel void reorder(global uint* cell_in,
                    global uint* cell_out,
                    global uint* counters,
                    global uint* prefixes,
                    local uint* l_counters,
                    local uint* l_prefixes,
                    configuration conf,
                    uint offset) {
  uint lid = get_local_id(0);
  uint gid = get_group_id(0);
  uint thread_grp = lid / conf.r_val;
  uint radix = conf.rpb * gid;
  uint groups = conf.gpb * conf.blocks;
  uint rc_offset = radix * groups;
  // erst abschließen der radix summierung
  for (uint i = 0 ; i < conf.rpb ; i++) {
    for (uint j = lid; j < groups; j += conf.tpb) {
      l_counters[i * groups + j] = counters[rc_offset + i * groups + j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  // Read radix prefixes to localMemory
  for (uint i = lid; i < conf.radices; i += conf.tpb) {
    l_prefixes[i] = prefixes[i];
  }
  // prefix sum on the prefixes in local memory
  barrier(CLK_LOCAL_MEM_FENCE);
  prefix_local(l_prefixes, conf.radices, conf.tpb);
  barrier(CLK_LOCAL_MEM_FENCE);
  // add prefix sum to the subcounters
  for (uint i = 0; i < conf.rpb; ++i) {
    for (uint j = lid; j < groups; j += conf.tpb) {
      l_counters[i * groups + j] += l_prefixes[radix + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  // Write back radices at the calculated offsets
  for (uint i = 0; i < conf.rpb; i++) {
    for (uint j = lid; j < groups; j += conf.tpb) {
      counters[rc_offset + i * groups + j] = l_counters[i * groups + j];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  uint active_block = gid * conf.gpb * conf.epg;
  uint active_counter = gid * conf.gpb;
  uint active_group = (lid / conf.r_val) * conf.epg;
  uint start = active_block + active_group + lid % conf.r_val;
  uint end = min(active_block + active_group + conf.epg, conf.size);
  for (uint i = start; i < end; i += conf.tpg) {
    uint bits = (cell_in[i] >> offset) & conf.mask;
    for (uint j = 0; j < conf.r_val; ++j) {
      if (lid % conf.r_val == j) {
        uint tmp = bits * groups + active_counter + thread_grp;
        cell_out[counters[tmp]] = cell_in[i];
        // TODO: do we need atomics or is the barrier enough?
        //atomic_inc(&counters[tmp]);
        ++counters[tmp];
      }
      barrier(CLK_GLOBAL_MEM_FENCE);
    }
  }
}

kernel void reorder_kv(global uint* cell_in,
                       global uint* cell_out,
                       global uint* value_in,
                       global uint* value_out,
                       global uint* counters,
                       global uint* prefixes,
                       local uint* l_counters,
                       local uint* l_prefixes,
                       configuration conf,
                       uint offset) {
  uint lid = get_local_id(0);
  uint gid = get_group_id(0);
  uint thread_grp = lid / conf.r_val;
  uint act_radix = conf.rpb * gid;
  uint groups = conf.gpb * conf.blocks;
  int rc_offset = act_radix * groups;
  // Finish the radix sum
  for (uint i = 0 ; i < conf.rpb ; i++) {
    for (uint blockidx = lid; blockidx < groups; blockidx += conf.tpb) {
      // The Num_Groups counters of the radix are read from global memory to
      // shared memory. Jeder Thread liest die Counter basierend auf der groupId
      // aus.
      l_counters[i * groups + blockidx]
        = counters[rc_offset + i * groups + blockidx];
    }
  }
  // Read radix prefixes to localMemory
  for (uint i = lid; i < conf.radices; i+= conf.tpb) {
    l_prefixes[i] = prefixes[i];
  }
  // build prefix sum over radix counters?
  // (seems to be the prefixes, not counters)
  barrier(CLK_GLOBAL_MEM_FENCE);
  prefix_local(l_prefixes, conf.radices, conf.tpb);
  barrier(CLK_GLOBAL_MEM_FENCE);
  // Add the sum of the radix to all subounter of the radixes
  for (uint i = 0; i < conf.rpb; i++) {
    for (uint j = lid; j < groups; j += conf.tpb) {
      l_counters[i * groups + j] += l_prefixes[act_radix + i];
    }
    // barrier(CLK_GLOBAL_MEM_FENCE);
  }
  // Write back the radixes to their respectve offers
  for (uint i = 0; i < conf.rpb; i++) {
    for (uint blockidx = lid; blockidx < groups; blockidx += conf.tpb) {
      // The Num_Groups counters of the radix are read from global memory to
      // shared memory. Jeder Thread liest die Counter basierend auf der
      // groupId aus
      counters[rc_offset + i * groups + blockidx]
        = l_counters[i * groups + blockidx];
    }
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  int active_block = gid * conf.gpb * conf.epg ;
  int active_counter = gid * conf.gpb  ;
  int active_group = (lid / conf.r_val) * conf.epg;
  uint idx = active_block + active_group +  lid % conf.r_val;
  //uint boundary = active_block + active_group + conf.epg;
  //boundary = (boundary < conf.size) ? boundary : conf.size;
  uint boundary = min(active_block + active_group + conf.epg, conf.size);
  for (; idx < boundary; idx += conf.tpg) {
    uint tmpRdx = (cell_in[idx] >> offset) & conf.mask;
    for (uint tmpIdx = 0; tmpIdx < conf.r_val; tmpIdx++) {
      if (lid % conf.r_val == tmpIdx) {
        uint tmp = tmpRdx * groups + active_counter + thread_grp;
        cell_out[counters[tmp]] = cell_in[idx];
        value_out[counters[tmp]] = value_in[idx];
        // TODO: is the increment ok, or should we use atomics?
        ++counters[tmp];
        //atomic_inc(&counters[tmp]);
      }
      barrier(CLK_GLOBAL_MEM_FENCE);
    }
  }
}


