/******************************************************************************
 * function: radix sort                                                       *
 * input:                                                                     *
 * output:                                                                    *
 ******************************************************************************/

 // Adapted from: https://github.com/Schw1ng/GPURadixSort

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

// The `global` thing here is just the memory placement, right?
inline void prefix_global(global uint* data, int len, int threads) {
  uint lid = get_local_id(0);
  int inc = 2;
  // reduce
  while (inc <= len) {
    int j = inc >> 1;
    for (int i = (j - 1) + (lid * inc); (i + inc) < len ; i += threads * inc)
      data[i + j] = data[i] + data[i + j];
    inc = inc << 1;
    barrier(CLK_GLOBAL_MEM_FENCE);
  }
  // Downsweep
  data[len-1] = 0;
  barrier(CLK_GLOBAL_MEM_FENCE);
  while (inc >= 2) {
    int j = inc >> 1;
    for (int i = (j - 1) + (lid * inc); (i + j) <= len; i += threads * inc) {
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
  uint grp = get_group_id(0);
  // Current thread group is based of the local id.
  uint thread_grp = lid / conf.r_val;
  // Startindex of the datablock that corresponds to the Threadblock.
  int active_block = grp * conf.gpb * conf.epg ;
  // Offset inside the block for the threadgroup of the current thread.
  int active_group = (lid / conf.r_val) * conf.epg;
  // Number of counters for each radix.
  uint groups = conf.blocks * conf.gpb;
  // Each Threadgroup has its own counter for each radix and is
  // calculating the offset based on current block / group.
  uint grp_offset = grp * conf.gpb;
  // Startindex for the current thread.
  uint start = active_block + active_group + lid % conf.r_val;
  //uint end = active_block + active_group + conf.epg;
  //end = (end > conf.size) ? conf.size : end;
  uint end = min(active_block + active_group + conf.epg, conf.size);
  for (;start < end; start += conf.tpg) {
    uint act_radix = (cell_in[start] >> offset) & conf.mask;
    // The following code ensures that the counters of each Threadgroup
    // are sequentially incremented.
    for (uint i = 0 ; i < conf.r_val; i++) {
      if (lid % conf.r_val == i) {
        uint start = (act_radix * groups) + grp_offset + thread_grp;
        // TODO: not a big fan of this, maybe use atomics?
        ++counters[start];
        //atomic_inc(&counters[start]);
      }
      barrier(CLK_GLOBAL_MEM_FENCE);
    }
  }
}

kernel void sum(global uint* cell_in,
                global volatile uint* counters,
                global uint* prefixes,
                local uint* counts,
                configuration conf,
                uint offset) {
  uint lid = get_local_id(0);
  uint grp = get_group_id(0);
  uint groups = conf.blocks * conf.gpb;
  uint act_radix = conf.rpb * grp;
  for (uint i = 0; i < conf.rpb; i++) {
    // The Num_Groups counters of the radix are read from global memory to
    // shared memory. Jeder Thread liest die Counter basierend auf der localid aus
    int k = 0;
    uint boarder = ((act_radix + 1) * groups);
    // boarder = (boarder > conf.blocks * conf.gpb)
    //         ? conf.blocks * conf.gpb : boarder;
    for (uint j = (act_radix * groups) + lid; j < boarder; j+= conf.tpb) {
      counts[lid + conf.tpb * k] = counters[j];
      ++k;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    // Die einzelnen RadixCounter sind nun in dem counts local Memory
    prefix_local(counts, groups, conf.tpb);
    barrier(CLK_GLOBAL_MEM_FENCE);
    // Gesamtprefix für den aktuellen radix berechnen
    if (lid == 1) {
      prefixes[act_radix]
        = counts[(groups) - 1] + counters[((act_radix + 1) * groups) - 1];
    }
    // Errechnete Prefixsumme zurück in den global memory schreiben
    barrier(CLK_GLOBAL_MEM_FENCE);
    k = 0;
    for (uint j = (act_radix * groups) + lid; j < ((act_radix + 1) * groups);
         j += conf.tpb) {
      counters[j] = counts[lid + conf.tpb * k];
      ++k;
    }
    ++act_radix;
  }
}

kernel void values(global uint* cell_in,
                   global uint* cell_out,
                   global uint* counters,
                   global uint* prefixes,
                   local uint* l_counters,
                   local uint* l_prefixes,
                   configuration conf,
                   uint offset) {
  uint lid = get_local_id(0);
  uint grp = get_group_id(0);
  uint thread_grp = lid / conf.r_val;
  uint act_radix = conf.rpb * grp;
  uint groups = conf.gpb * conf.blocks;
  int rc_offset = act_radix * groups;
  // erst abschließen der radix summierung
  for (uint i = 0 ; i < conf.rpb ; i++) {
    for (uint idx = lid; idx < groups; idx += conf.tpb) {
      // The Num_Groups counters of the radix are read from global memory
      // to shared memory. Jeder Thread liest die Counter basierend auf der
      // groupId aus
      l_counters[i * groups + idx] = counters[rc_offset + i * groups + idx];
    }
  }
  // Read radix prefixes to localMemory
  for (uint i = lid; i < conf.radices; i += conf.tpb) {
    l_prefixes[i] = prefixes[i];
  }
  // Präfixsumme über die RadixCounter bilden.
  barrier(CLK_GLOBAL_MEM_FENCE);
  prefix_local(l_prefixes, conf.radices, conf.tpb);
  barrier(CLK_GLOBAL_MEM_FENCE);
  // Die Präfixsumme des Radixe auf alle subcounter der radixes addieren
  for (uint i = 0; i < conf.rpb; i++) {
    for (uint j = lid; j < groups; j += conf.tpb) {
      l_counters[i * groups + j] += l_prefixes[act_radix + i];
    }
    // barrier(CLK_GLOBAL_MEM_FENCE);
  }
  // Zurückschreiben der Radixe mit entsprechedem offset.
  for (uint i = 0; i < conf.rpb; i++) {
    for (uint idx = lid; idx < groups; idx += conf.tpb) {
      // The Num_Groups counters of the radix are read from global memory
      // to shared memory. Jeder Thread liest die Counter basierend auf
      // der groupId aus
      counters[rc_offset + i * groups + idx] = l_counters[i * groups + idx];
    }
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  int active_block = grp * conf.gpb * conf.epg;
  int active_counter = grp * conf.gpb;
  int active_group = (lid / conf.r_val) * conf.epg;
  uint idx = active_block + active_group + lid % conf.r_val;
  //uint boundary = active_block + active_group + conf.epg;
  //boundary = (boundary < conf.size) ? boundary : conf.size;
  uint boundary = min(active_block + active_group + conf.epg, conf.size);
  for (; idx < boundary; idx += conf.tpg) {
    uint tmpRdx = (cell_in[idx] >> offset) & conf.mask;
    for (uint tmpIdx = 0; tmpIdx < conf.r_val; tmpIdx++) {
      if (lid % conf.r_val == tmpIdx) {
        uint idx = tmpRdx * groups + active_counter + thread_grp;
        cell_out[counters[idx]] = cell_in[idx];
        ++counters[idx];
      }
      barrier(CLK_GLOBAL_MEM_FENCE);
    }
  }
}

kernel void values_by_keys(global uint* cell_in,
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
  uint grp = get_group_id(0);
  uint thread_grp = lid / conf.r_val;
  uint act_radix = conf.rpb * grp;
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
  int active_block = grp * conf.gpb * conf.epg ;
  int active_counter = grp * conf.gpb  ;
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


