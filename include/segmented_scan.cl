
// For optimizations regarding bank conflicts, look at:
// - http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
// - Scan Primitives for GPUs (Sengupta et al)

/// Global segmented scan, phase 1. From: 
/// - Scan Primitives for GPUs (Sengupta et al)
kernel void upsweep(global uint* restrict data,
                    global uint* restrict heads,
                    global uint* restrict last_data,
                    global uint* restrict last_head,
                    global uint* restrict tree,
                    local uint* d,
                    local uint* h,
                    uint len) {
  const uint thread = get_local_id(0);
  const uint block = get_group_id(0);
  const uint threads_per_block = get_local_size(0);
  const uint elements_per_block = threads_per_block * 2;
  const uint global_offset = block * elements_per_block;
  const uint n = elements_per_block;

  const uint e = 2 * thread;      // even
  const uint o = 2 * thread + 1;  // odd
  const uint ge = 2 * get_global_id(0);     // global_offset + e
  const uint go = 2 * get_global_id(0) + 1; // global_offset + o
  d[e] = (e < len) ? data [ge] : 0;
  d[o] = (o < len) ? data [go] : 0;
  h[e] = (e < len) ? heads[ge] : 0;
  h[o] = (o < len) ? heads[go] : 0;
  /*
  int depth = 1 + (int) log2((float) n);
  for (int i = 0; i < depth; i++) {
    barrier(CLK_LOCAL_MEM_FENCE);
    int mask = (0x1 << i) - 1;
    if ((thread & mask) == mask) {
      int offset = (0x1 << i);
      int bi = o;
      int ai = bi - offset;
      if (!h[bi]) {
        d[bi] += d[ai];
      }
      h[bi] = h[bi] | h[ai];
    }
  }
  */
  uint offset = 1;
  for (uint i = n >> 1; i > 0; i >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thread < i) {
      int ai = offset * (e + 1) - 1;
      int bi = offset * (o + 1) - 1;
      d[bi] += (h[bi] == 0) ? d[ai] : 0;
      h[bi] |= h[ai];
    }
    offset <<= 1;
  }
  if (thread == 0) {
    tree[block] = heads[global_offset];
    last_data[block] = d[n - 1];
    last_head[block] = h[n - 1];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (ge < len) {
    data[ge] = d[e];
    heads[ge] = h[e];
  }
  if (go < len) {
    data[go] = d[o];
    heads[go] = h[o];
  }
}

/// Global exclusive scan, phase 2.
kernel void block_scan(global uint* restrict data,
                       global uint* restrict heads,
                       global uint* restrict tree,
                       uint len) {
  local uint d[2048];
  local uint h[2048];
  local uint t[2048];
  const uint thread = get_local_id(0);
  const uint n = 2048;
  uint offset = 1;
  const uint e = 2 * thread;
  const uint o = 2 * thread + 1;
  d[e] = (e < len) ? data[e] : 0;
  d[o] = (o < len) ? data[o] : 0;
  h[e] = (e < len) ? heads[e] : 0;
  h[o] = (o < len) ? heads[o] : 0;
  t[e] = (e < len) ? tree[e] : 0;
  t[o] = (o < len) ? tree[o] : 0;
  // build sum in place up the tree
  for (uint i = n >> 1; i > 0; i >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thread < i) {
      int ai = offset * (e + 1) - 1;
      int bi = offset * (o + 1) - 1;
      d[bi] += (h[bi] == 0) ? d[ai] : 0;
      h[bi] |= h[ai];
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
      int ai = offset * (e + 1) - 1;
      int bi = offset * (o + 1) - 1;
      uint tmp = d[ai];
      d[ai] = d[bi];
      if (t[ai + 1] == 1)
        d[bi] = 0;
      else if (h[ai] == 1)
        d[bi] = tmp;
      else
        d[bi] += tmp;
      h[ai] = 0;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (e < len) {
    data[e] = d[e];
    heads[e] = h[e];
  }
  if (o < len) {
    data[o] = d[o];
    heads[o] = h[o];
  }
}

// Global segmented scan phase 3.
kernel void downsweep(global uint* restrict data,
                      global uint* restrict heads,
                      global uint* restrict last_data,
                      global uint* restrict last_head,
                      global uint* restrict tree,
                      local uint* d,
                      local uint* h,
                      local uint* t,
                      uint len) {
  const uint thread = get_local_id(0);
  const uint block = get_group_id(0);
  const uint threads_per_block = get_local_size(0);
  const uint elements_per_block = threads_per_block * 2;
  const uint global_offset = block * elements_per_block;
  const uint n = elements_per_block;

  const uint e = 2 * thread;
  const uint o = 2 * thread + 1;
  const uint ge = 2 * get_global_id(0);     // global_offset + e
  const uint go = 2 * get_global_id(0) + 1; // global_offset + o
  d[e] = (e < len) ? data[ge] : 0;
  d[o] = (o < len) ? data[go] : 0;
  h[e] = (e < len) ? heads[ge] : 0;
  h[o] = (o < len) ? heads[go] : 0;
  t[e] = (e < len) ? tree[ge] : 0;
  t[o] = (o < len) ? tree[go] : 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (thread == 0) {
    d[n - 1] = last_data[block];
    h[n - 1] = last_head[block];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
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
      } else if (h[ai]) {
        d[bi] = tmp;
      } else {
        d[bi] += tmp;
      }
      h[ai] = 0;
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
      int ai = offset * (e + 1) - 1;
      int bi = offset * (o + 1) - 1;
      uint tmp = d[ai];
      d[ai] = d[bi];
      if (t[ai + 1] == 1)
        d[bi] = 0;
      else if (h[ai] == 1)
        d[bi] = tmp;
      else
        d[bi] += tmp;
      h[ai] = 0;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (ge < len) data[ge] = d[e];
  if (go < len) data[go] = d[o];
}
