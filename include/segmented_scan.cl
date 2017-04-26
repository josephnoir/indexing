
// For optimizations regarding bank conflicts, look at:
// - http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
// - Scan Primitives for GPUs (Sengupta et al)

/// Global segmented scan, phase 1. From: 
/// - Scan Primitives for GPUs (Sengupta et al)
/// Arguments:
/// data      --> data to scan
/// part      --> partition flags for segments
/// last_data --> last data entry of each block after upsweep
/// last_part --> last 
/// tree      --> save the first flags for later use
kernel void upsweep(global uint* restrict data,
                    global uint* restrict part,
                    global uint* restrict last_data,
                    global uint* restrict last_part,
                    global uint* restrict tree,
                    local uint* d,
                    local uint* p,
                    uint len) {
  const uint x = get_global_id(0);
  const uint thread = get_local_id(0);
  const uint block = get_group_id(0);
  const uint threads_per_block = get_local_size(0);
  const uint elements_per_block = threads_per_block * 2;
  const uint global_offset = block * elements_per_block;
  const uint n = elements_per_block;
  const uint e = 2 * thread;      // even
  const uint o = 2 * thread + 1;  // odd
  const uint ge = 2 * x;          // was: global_offset + e
  const uint go = 2 * x + 1;      // was: global_offset + o
  d[e] = (e < len) ? data [ge] : 0;
  d[o] = (o < len) ? data [go] : 0;
  p[e] = (e < len) ? part[ge] : 0;
  p[o] = (o < len) ? part[go] : 0;
  /*
  int depth = 1 + (int) log2((float) n);
  for (int i = 0; i < depth; i++) {
    barrier(CLK_LOCAL_MEM_FENCE);
    int mask = (0x1 << i) - 1;
    if ((thread & mask) == mask) {
      int offset = (0x1 << i);
      int bi = o;
      int ai = bi - offset;
      if (!p[bi]) {
        d[bi] += d[ai];
      }
      p[bi] = p[bi] | p[ai];
    }
  }
  */
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
  if (thread == 0)
    tree[block] = part[e];
  //barrier(CLK_LOCAL_MEM_FENCE);
  if (thread == threads_per_block - 1) {
    last_data[block] = d[o];
    last_part[block] = p[o];
  }
  if (ge < len) {
    data[ge] = d[e];
    part[ge] = p[e];
  }
  if (go < len) {
    data[go] = d[o];
    part[go] = p[o];
  }
}

/// Global exclusive scan, phase 2.
/// Arguments:
/// data      --> data to scan
/// part      --> partition flags for segments
/// tree      --> ...
kernel void block_scan(global uint* restrict data,
                       global uint* restrict part,
                       global uint* restrict tree,
                       uint len) {
  local uint d[2048];
  local uint p[2048];
  local uint t[2048];
  const uint thread = get_local_id(0);
  const uint n = 2048;
  const uint e = 2 * thread;
  const uint o = 2 * thread + 1;
  d[e] = (e < len) ? data[e] : 0;
  d[o] = (o < len) ? data[o] : 0;
  p[e] = (e < len) ? part[e] : 0;
  p[o] = (o < len) ? part[o] : 0;
  t[e] = (e < len) ? tree[e] : 0;
  t[o] = (o < len) ? tree[o] : 0;
  // build sum in place up the tree
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
  // traverse down tree & build scan
  for (uint i = 1; i < n; i <<= 1) {
    offset >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thread < i) {
      const uint ai = offset * (e + 1) - 1;
      const uint bi = offset * (o + 1) - 1;
      const uint tmp = d[ai];
      d[ai] = d[bi];
      if (t[ai + 1] == 1)
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
}

/// Global segmented scan phase 3
/// Arguments:
/// data      --> data to scan
/// part      --> partition flags for segments
/// last_data --> last data entry of each block after upsweep
/// last_part --> last 
/// tree      --> save the first flags for later use.
kernel void downsweep(global uint* restrict data,
                      global uint* restrict part,
                      global uint* restrict last_data,
                      global uint* restrict last_part,
                      global uint* restrict tree,
                      local uint* d,
                      local uint* p,
                      local uint* t,
                      uint len) {
  const uint x = get_global_id(0);
  const uint thread = get_local_id(0);
  const uint block = get_group_id(0);
  const uint threads_per_block = get_local_size(0);
  const uint elements_per_block = threads_per_block * 2;
  const uint global_offset = block * elements_per_block;
  const uint n = elements_per_block;
  const uint e = 2 * thread;
  const uint o = 2 * thread + 1;
  const uint ge = 2 * x;     // global_offset + e
  const uint go = 2 * x + 1; // global_offset + o
  // Load data into local memory
  d[e] = (e < len) ? data[ge] : 0;
  d[o] = (o < len) ? data[go] : 0;
  p[e] = (e < len) ? part[ge] : 0;
  p[o] = (o < len) ? part[go] : 0;
  t[e] = (e < len) ? tree[ge] : 0;
  t[o] = (o < len) ? tree[go] : 0;
  // Load results from block scan
  if (thread == threads_per_block - 1) {
    d[o] = last_data[block];
    p[o] = last_part[block];
  }
  // downsweep
  /*
  int depth = (int) log2((float) n);
  for (int i = depth; i > -1; i--) {
    barrier(CLK_LOCAL_MEM_FENCE);
    int mask = (0x1 << i) - 1;
    if ((thread & mask) == mask) {
      int offset = (0x1 << i);
      int bi = o;
      int ai = bi - offset;
      int tmp = d[ai];
                d[ai] = d[bi];
      if (t[ai + 1]) {
        d[bi] = 0;
      } else if (p[ai]) {
        d[bi] = tmp;
      } else {
        d[bi] += tmp;
      }
      p[ai] = 0;
    }
  }
  */
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
      if (t[ai + 1] == 1)
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
}
