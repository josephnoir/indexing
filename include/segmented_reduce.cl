/******************************************************************************
 * Copyright (C) 2017                                                         *
 * Raphael Hiesgen <raphael.hiesgen (at) haw-hamburg.de>                      *
 *                                                                            *
 * Distributed under the terms and conditions of the BSD 3-Clause License.    *
 *                                                                            *
 * If you did not receive a copy of the license files, see                    *
 * http://opensource.org/licenses/BSD-3-Clause and                            *
 ******************************************************************************/

// For optimizations regarding bank conflicts, see:
// - http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
// - Scan Primitives for GPUs (Sengupta et al)

/// Global segmented scan, phase 1. From: 
/// - Scan Primitives for GPUs (Sengupta et al)
/// Arguments:
/// data      --> data to scan
/// part      --> partition flags for segments
/// last_data --> last data entry of each block after upsweep
/// last_part --> last 
/// flag      --> save the first flags for later use
kernel void upsweep(global uint* restrict data,
                    global uint* restrict part,
                    global uint* restrict flag,
                    global uint* restrict last_data,
                    global uint* restrict last_part,
                    global uint* restrict last_flag,
                    local uint* d,
                    local uint* p,
                    uint len) {
  // TODO: implement
  /*
  const uint x = get_global_id(0);
  const uint thread = get_local_id(0);
  const uint block = get_group_id(0);
  const uint threads_per_block = get_local_size(0);
  const uint elements_per_block = threads_per_block * 2;
  const uint n = elements_per_block;
  const uint e = 2 * thread;      // even
  const uint o = 2 * thread + 1;  // odd
  const uint ge = 2 * x;
  const uint go = 2 * x + 1;
  d[e] = (ge < len) ? data[ge] : 0;
  d[o] = (go < len) ? data[go] : 0;
  p[e] = (ge < len) ? part[ge] : 0;
  p[o] = (go < len) ? part[go] : 0;
  uint offset = 1;
  for (uint i = n >> 1; i > 0; i >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thread < i) {
      const uint ai = offset * (e + 1) - 1;
      const uint bi = offset * (o + 1) - 1;
      d[bi] += (p[bi] == 0) ? d[ai] : 0;
      p[bi] |= p[ai];
    }
    offset <<= 1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (thread == 0)
    last_flag[block] = flag[ge];
  //barrier(CLK_LOCAL_MEM_FENCE);
  if (thread == threads_per_block - 1) {
    last_data[block] = d[n - 1];
    last_part[block] = p[n - 1];
  }
  if (ge < len) {
    data[ge] = d[e];
    part[ge] = p[e];
  }
  if (go < len) {
    data[go] = d[o];
    part[go] = p[o];
  }
  */
}

/// Global exclusive scan, phase 2.
/// Arguments:
/// data      --> data to scan
/// part      --> partition flags for segments
/// flag      --> ...
kernel void block_reduce(global uint* restrict data,
                         global uint* restrict part,
                         global uint* restrict flag,
                         uint len) {
  // TODO: implement
  /*
  local uint d[1024];
  local uint p[1024];
  local uint f[1024];
  const uint thread = get_local_id(0);
  const uint threads_per_block = get_local_size(0);
  const uint elements_per_block = 2 * threads_per_block;
  const uint n = elements_per_block;
  const uint e = 2 * thread;
  const uint o = 2 * thread + 1;
  d[e] = (e < len) ? data[e] : 0;
  d[o] = (o < len) ? data[o] : 0;
  p[e] = (e < len) ? part[e] : 0;
  p[o] = (o < len) ? part[o] : 0;
  f[e] = (e < len) ? flag[e] : 0;
  f[o] = (o < len) ? flag[o] : 0;
  // build sum in place up the flag
  uint offset = 1;
  for (uint i = n >> 1; i > 0; i >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thread < i) {
      const int ai = offset * (e + 1) - 1;
      const int bi = offset * (o + 1) - 1;
      d[bi] += (p[bi] == 0) ? d[ai] : 0;
      p[bi] |= p[ai];
    }
    offset <<= 1;
  }
  if (thread == 0)
    d[n - 1] = 0;
  // traverse down flag & build scan
  for (uint i = 1; i < n; i <<= 1) {
    offset >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thread < i) {
      const uint ai = offset * (e + 1) - 1;
      const uint bi = offset * (o + 1) - 1;
      const uint tmp = d[ai];
      d[ai] = d[bi];
      if (f[ai + 1] == 1)
        d[bi] = 0;
      else if (p[ai] == 1)
        d[bi] = tmp;
      else
        d[bi] += tmp;
      p[ai] = 0;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (e < len) {
    data[e] = d[e];
    part[e] = p[e];
  }
  if (o < len) {
    data[o] = d[o];
    part[o] = p[o];
  }
  */
}

/// Global segmented scan phase 3
/// Arguments:
/// data      --> data to scan
/// part      --> partition flags for segments
/// last_data --> last data entry of each block after upsweep
/// last_part --> last 
/// flag      --> save the first flags for later use.
kernel void downsweep(global uint* restrict data,
                      global uint* restrict part,
                      global uint* restrict flag,
                      global uint* restrict last_data,
                      global uint* restrict last_part,
                      local uint* d,
                      local uint* p,
                      local uint* f,
                      uint len) {
  // TODO: implement
  /*
  const uint x = get_global_id(0);
  const uint thread = get_local_id(0);
  const uint block = get_group_id(0);
  const uint threads_per_block = get_local_size(0);
  const uint elements_per_block = threads_per_block * 2;
  const uint n = elements_per_block;
  const uint e = 2 * thread;
  const uint o = 2 * thread + 1;
  const uint ge = 2 * x;
  const uint go = 2 * x + 1;
  // Load data into local memory
  d[e] = (ge < len) ? data[ge] : 0;
  d[o] = (go < len) ? data[go] : 0;
  p[e] = (ge < len) ? part[ge] : 1;
  p[o] = (go < len) ? part[go] : 1;
  f[e] = (ge < len) ? flag[ge] : 0;
  f[o] = (go < len) ? flag[go] : 0;
  // Load results from block scan
  if (thread == threads_per_block - 1) {
    d[n - 1] = last_data[block];
    p[n - 1] = last_part[block];
  }
  // downsweep
  uint offset = 1;
  for (uint i = n >> 1; i > 0; i >>= 1)
    offset <<= 1;
  for (uint i = 1; i < n; i *= 2) {
    offset >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thread < i) {
      const int ai = offset * (e + 1) - 1;
      const int bi = offset * (o + 1) - 1;
      const uint tmp = d[ai];
      d[ai] = d[bi];
      if (f[ai + 1] == 1)
        d[bi] = 0;
      else if (p[ai] == 1)
        d[bi] = tmp;
      else
        d[bi] += tmp;
      p[ai] = 0;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // Write results back to global memory
  if (ge < len) data[ge] = d[e];
  if (go < len) data[go] = d[o];
  */
}
