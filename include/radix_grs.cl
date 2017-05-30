/******************************************************************************
 * function: radix sort                                                       *
 * input:                                                                     *
 * output:                                                                    *
 ******************************************************************************/

// Adapted from paper: GRS - GPU Radix Sort For Multifield Records


kernel void histogram(global uint4* restrict keys_in4,
                      global uint4* restrict ranks4,
                      global uint*  restrict counters,
                      uint startbit) {
  const uint t = 1024;                // tile size
  const uint tid = get_local_id(0);   // thread id
  const uint bid = get_group_id(0);   // block id
  const uint itrs = t / 32;

  local uint4 s_keys4[512];

  for (uint i = 0; i < itrs; ++i) {
    // Read keys from global memory
    uint key_offset = 0;
    uint start_tile = (bid * 64 + tid / 8) * (t / 4);
    uint key_pos = key_offset + tid % 8;
    uint s_key_pos = (tid / 8) * 8 + (((tid / 8) % 8) + (tid % 8)) % 8;
    uint tile_size8 = 8 * (t / 4);
    uint tid4 = tid * 4;
    for (int j = 0; j < 16; ++j) {
      s_hist[tid * 16 + i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint cur_tile_id = start_tile; // start_tile_id
    for (int j = 0; j < 8; ++j) {
      s_keys4[s_key_pos + i * 64] = keys_in4[key_pos + start_tile];
      cur_tile_id += tile_size8;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // process an element in s_hist4
    uint4 p4, r4;
    for (int j = 0; j < 8; ++j) {
      p4 = s_keys4[tid4 + (i + tid) % 8];
      r4.x = s_hist[((p4.x >> startbit) & 0xF) * 64 + tid]++;
      r4.y = s_hist[((p4.y >> startbit) & 0xF) * 64 + tid]++;
      r4.z = s_hist[((p4.z >> startbit) & 0xF) * 64 + tid]++;
      r4.w = s_hist[((p4.w >> startbit) & 0xF) * 64 + tid]++;
      s_keys4[tid4 + (i + tid) % 8] = r4;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // Write ranks to global mem
    cur_tile_id = start_tile; // start_tile_id
    for (int j = 0; j < 8; ++j) {
      ranks4[key_offset + key_pos + start_tile] = s_keys4[s_key_pos + i * 64]; // start_tile_id
      cur_tile_id += tile_size8;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  // write histogram to global memory
  uint global_tid = bid * 64 + tid;
  for (uint j = 0; j < 16; ++j) {
    counters[i * n_tiles + global_tid] = s_hist[i * 64 + tid];
  }
}

kernel void scan() {
  // prefix sum on ...
}

kernel void reorder(keys_out, recs_out, keys_in4, recs_in4, counters, counters_sum, ranks4) {
  int thread = get_local_id(0);
  int block = get_group_id(0);
  local s_tile_cnt[16];
  local s_g_offset[16];
  local int s_keys[t];
  local int s_fields[t];
  int4 k4;
  int4 r4;
  // read the histograms from global mem
  if (thread < 16) {
    s_tile_cnt[thread] = counters[thread * n_tiles + block];
    s_g_offset[thread] = counters_sum[thread * n_tiles + block];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  s_tile_cnt = warp_scan(s_tile_cnt);
  barrier(CLK_LOCAL_MEM_FENCE);
  k4 = keys_in4[block * (t / 4) + thread];
  r4 = ranks4[block * (t / 4) + thread];
  r4.x = r4.x + s_tile_cnt[(k4.x >> startbit) & 0xF];
  s_keys[r4.x] = k4.x;
  barrier(CLK_LOCAL_MEM_FENCE);
  radix = (s_keys[thread] >> startbit) & 0xF;
  global_offset
}

