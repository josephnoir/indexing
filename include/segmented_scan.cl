
// For optimizations regarding bank conflicts, look at:
// - http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
// - Scan Primitives for GPUs (Sengupta et al)

/// Global segmented scan, phase 1. From: 
/// - Scan Primitives for GPUs (Sengupta et al)
kernel void upsweep(global uint* restrict data,
                    global uint* restrict heads,
                    global uint* restrict increments,
                    global uint* restrict increment_heads,
                    global uint* restrict tree,
                    local uint* tmp_data,
                    local uint* tmp_heads,
                    uint len) {
  const uint thread = get_local_id(0);
  const uint block = get_group_id(0);
  const uint threads_per_block = get_local_size(0);
  const uint elements_per_block = threads_per_block * 2;
  const uint global_offset = block * elements_per_block;
  const uint n = elements_per_block;
  printf("thread %d in block %d\n", thread, block);

  uint offset = 1;
  uint even = 2 * thread;
  uint odd = 2 * thread + 1;
  tmp_data[even] = (even < len) ? data[global_offset + (even)] : 0;
  tmp_data[odd] = (odd < len) ? data[global_offset + (odd)] : 0;
  tmp_heads[even] = (even < len) ? heads[global_offset + (even)] : 0;
  tmp_heads[odd] = (odd < len) ? heads[global_offset + (odd)] : 0;
  // build sum in place up the tree
  for (uint d = n >> 1; d > 0; d >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thread < d) {
      int ai = offset * (even + 1) - 1;
      int bi = offset * (odd + 1) - 1;
      // change to segmented scan operator
      // eval flag ...
      tmp_data[bi] += tmp_heads[bi] ? 0 : tmp_data[ai];
      // ... before changing it!
      tmp_heads[bi] |= tmp_heads[ai];
    }
    offset <<= 1; // offset *= 2;
  }
  if (even < len) {
    data[global_offset + even] = tmp_data[even];
    heads[global_offset + even] = tmp_heads[even];
  }
  if (odd < len) {
    data[global_offset + odd] = tmp_data[odd];
    heads[global_offset + odd] = tmp_heads[odd];
  }
  if (thread == 0) {
    increments[block] = tmp_data[n - 1];
    increment_heads[block] = tmp_heads[n - 1];
    tree[block] = tmp_heads[0];
    // tmp_data[n - 1] = 0;
  }
}

/// Global exclusive scan, phase 2.
kernel void block_scan(global uint* restrict data,
                       global uint* restrict heads,
                       global uint* restrict tree,
                       uint len) {
  /*
  local uint tmp_data[2048];
  local uint tmp_heads[2048];
  const uint thread = get_local_id(0);
  const uint n = 2048;
  uint offset = 1;
  uint even = 2 * thread;
  uint odd = 2 * thread + 1;
  tmp_data[even] = (even < len) ? data[even] : 0;
  tmp_data[odd] = (odd < len) ? data[odd] : 0;
  tmp_heads[even] = (even < len) ? heads[even] : 1;
  tmp_heads[odd] = (odd < len) ? heads[odd] : 1;
  // build sum in place up the tree
  for (uint d = n >> 1; d > 0; d >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thread < d) {
      int ai = offset * (even + 1) - 1;
      int bi = offset * (odd + 1) - 1;
      // change to segmented scan operator
      // eval flag ...
      tmp_data[bi] += tmp_heads[bi] ? 0 : tmp_data[ai];
      // ... before changing it!
      tmp_heads[bi] |= tmp_heads[ai];
    }
    offset <<= 1;
  }
  if (thread == 0) {
    tmp_data[n - 1] = 0;
  }
  // traverse down tree & build scan
  for (uint d = 1; d < n; d <<= 1) {
    offset >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thread < d) {
      int ai = offset * (even + 1) - 1;
      int bi = offset * (odd + 1) - 1;
      uint next = (offset * (even + 1));
      uint t = tmp_data[ai];
      uint old = tmp_data[bi]; // TODO: delete this
      tmp_data[ai] = tmp_data[bi];
      // modified for segmented scan
      if (tree[next] == 1) // what if next >= len
        tmp_data[bi] = 0;
      else if (tmp_heads[ai] == 1)
        tmp_data[bi] = t;
      else
        tmp_data[bi] += t;
      tmp_heads[ai] = 0;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (even < len) {
    data[even] = tmp_data[even];
    heads[even] = tmp_heads[even];
  }
  if (odd < len) {
    data[odd] = tmp_data[odd];
    heads[odd] = tmp_heads[odd];
  }
  */
}

// Global segmented scan phase 3.
kernel void downsweep(global uint* restrict data,
                      global uint* restrict heads,
                      global uint* restrict increments,
                      global uint* restrict increment_heads,
                      global uint* restrict tree,
                      local uint* tmp_data,
                      local uint* tmp_heads,
                      uint len) {
  const uint thread = get_local_id(0);
  const uint block = get_group_id(0);
  const uint threads_per_block = get_local_size(0);
  const uint elements_per_block = threads_per_block * 2;
  const uint global_offset = block * elements_per_block;
  const uint n = elements_per_block;

  uint offset = 1;
  uint even = 2 * thread;
  uint odd = 2 * thread + 1;
  tmp_data[even] = (even < len) ? data[global_offset + (even)] : 0;
  tmp_data[odd] = (odd < len) ? data[global_offset + (odd)] : 0;
  tmp_heads[even] = (even < len) ? heads[global_offset + (even)] : 0;
  tmp_heads[odd] = (odd < len) ? heads[global_offset + (odd)] : 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (thread == 0) {
    printf("Writing %d / %d to %d\n", increments[block], increment_heads[block], (n - 1));
    tmp_data[n - 1] = increments[block];
    tmp_heads[n - 1] = increment_heads[block];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // probably always the <= power of two
  for (uint d = n >> 1; d > 0; d >>= 1)
    offset <<= 1;
  // traverse down tree & build scan
  printf("offset is %d, n is %d\n", offset, n);
  for (uint d = 1; d < n; d *= 2) {
    offset >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thread < d) {
      int ai = offset * (even + 1) - 1;
      int bi = offset * (odd + 1) - 1;
      uint next = global_offset + (offset * (even + 1));
      uint t = tmp_data[ai];
      tmp_data[ai] = tmp_data[bi];
      // modified for segmented scan
      if (heads[next] == 1) // what if next >= len
        tmp_data[bi] = 0;
      else if (tmp_heads[ai] == 1)
        tmp_data[bi] = t;
      else
        tmp_data[bi] += t;
      tmp_heads[ai] = 0;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (even < len) {
    data[global_offset + even] = tmp_data[even];
    heads[global_offset + even] = tmp_heads[even];
  }
  if (odd < len) {
    data[global_offset + odd] = tmp_data[odd];
    heads[global_offset + odd] = tmp_heads[odd];
  }
}
