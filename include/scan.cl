
// For optimizations regarding bank conflicts, look at:
// http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
// TODO: Get efficient scan to work with arbitrary n > 0
// scan consisting of upsweep, null_last and downsweep does not work!!!

kernel void lazy_scan(global uint* restrict in, global uint* restrict out,
                      uint len) {
  uint idx = get_global_id(0);
  if (idx == 0) {
    out[0] = 0;
    for (uint i = 1; i < len; ++i)
      out[i] = out[i - 1] + in[i - 1];
  }
}

/// Prefix scan in local memory, tmp size should be == n.
/// Requires n/2 threads. N must be a power of 2.
/// From: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
kernel void prescan(global uint* restrict data, local uint* tmp, uint n) {
  // extern __shared__ float tmp[];  // allocated on invocation  
  uint thread = get_local_id(0); // threadIdx.x;
  uint offset = 1;
  // A (2 lines) --> load input into shared memory
  tmp[2 * thread    ] = data[2 * thread    ];
  tmp[2 * thread + 1] = data[2 * thread + 1];
  // build sum in place up the tree
  for (uint d = n >> 1; d > 0; d >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();
    if (thread < d) {
      // B (2 lines)
      int ai = offset * (2 * thread + 1) - 1;
      int bi = offset * (2 * thread + 2) - 1;
      tmp[bi] += tmp[ai];
    }
    offset *= 2;
  }
  // C (2 lines) --> clear the last element
  if (thread == 0)
    tmp[n - 1] = 0;
  // traverse down tree & build scan
  for (uint d = 1; d < n; d *= 2) {
    offset >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();
    if (thread < d) {
      // D (2 lines)
      int ai = offset * (2 * thread + 1) - 1;
      int bi = offset * (2 * thread + 2) - 1;
      uint t = tmp[ai];
      tmp[ai] = tmp[bi];
      tmp[bi] += t;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // E (2 line) --> write results to device memory
  data[2 * thread    ] = tmp[2 * thread    ];
  data[2 * thread + 1] = tmp[2 * thread + 1];
}


#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> (NUM_BANKS + (n)) >> (2 * LOG_NUM_BANKS))

/// Prefix scan in local memory, tmp size should be == n.
/// Requires n/2 threads. N must be a power of 2.
/// Optimized for memory bank comflicts.
/// From: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
kernel void prescan2(global uint* restrict data, local uint* tmp, uint n) {
  // extern __shared__ float tmp[];  // allocated on invocation  
  uint thread = get_local_id(0); // threadIdx.x;
  uint offset = 1;
  // New A (6 lines) --> load input into shared memory 
  uint ai = thread;
  uint bi = thread + (n / 2);
  uint bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  uint bankOffsetB = CONFLICT_FREE_OFFSET(bi);
  tmp[ai + bankOffsetA] = data[ai];
  tmp[bi + bankOffsetB] = data[bi];
  // build sum in place up the tree
  for (uint d = n >> 1; d > 0; d >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();
    if (thread < d) {
      // New B (4 lines)
      uint ai = offset * (2 * thread + 1) - 1;
      uint bi = offset * (2 * thread + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);
      tmp[bi] += tmp[ai];
    }
    offset *= 2;
  }
  // New C (2 lines) --> clear the last element
  if (thread == 0)
    tmp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
  // traverse down tree & build scan
  for (uint d = 1; d < n; d *= 2) {
    offset >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();
    if (thread < d) {
      // New D (4 lines)
      uint ai = offset * (2 * thread + 1) - 1;
      uint bi = offset * (2 * thread + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);
      uint t = tmp[ai];
      tmp[ai] = tmp[bi];
      tmp[bi] += t;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // New E (2 lines) --> write results to device memory
  data[ai] = tmp[ai + bankOffsetA];
  data[bi] = tmp[bi + bankOffsetB];
}


/// Global exclusive scan, phase 1.
/// From: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
kernel void es_phase_1(global uint* restrict data,
                       global uint* restrict increments,
                       local uint* tmp, uint len) {
  const uint thread = get_local_id(0);
  const uint block = get_group_id(0);
  const uint threads_per_block = get_local_size(0);
  const uint elements_per_block = threads_per_block * 2;
  const uint global_offset = block * elements_per_block;
  const uint n = elements_per_block;

  uint offset = 1;
  // A (2 lines) --> load input into shared memory
  tmp[2 * thread    ] = (2 * thread     < len) ? data[global_offset + (2 * thread    )] : 0;
  tmp[2 * thread + 1] = (2 * thread + 1 < len) ? data[global_offset + (2 * thread + 1)] : 0;
  // build sum in place up the tree
  for (uint d = n >> 1; d > 0; d >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();
    if (thread < d) {
      // B (2 lines)
      int ai = offset * (2 * thread + 1) - 1;
      int bi = offset * (2 * thread + 2) - 1;
      tmp[bi] += tmp[ai];
    }
    offset *= 2;
  }
  // C (2 lines) --> clear the last element
  if (thread == 0) {
    increments[block] = tmp[n - 1];
    tmp[n - 1] = 0;
  }
  // traverse down tree & build scan
  for (uint d = 1; d < n; d *= 2) {
    offset >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();
    if (thread < d) {
      // D (2 lines)
      int ai = offset * (2 * thread + 1) - 1;
      int bi = offset * (2 * thread + 2) - 1;
      uint t = tmp[ai];
      tmp[ai] = tmp[bi];
      tmp[bi] += t;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // E (2 line) --> write results to device memory
  if (2 * thread < len)
    data[global_offset + (2 * thread    )] = tmp[2 * thread    ];
  if (2 * thread + 1 < len)
    data[global_offset + (2 * thread + 1)] = tmp[2 * thread + 1];
}

/// Global exclusive scan, phase 2.
kernel void es_phase_2(global uint* restrict data,
                       global uint* restrict increments,
                       local uint* tmp, uint len) {
  (void) data; // just to keep the arguments similar
  const uint thread = get_local_id(0);
  const uint block = get_group_id(0);
  const uint threads_per_block = get_local_size(0);
  const uint elements_per_block = threads_per_block * 2;
  const uint global_offset = block * elements_per_block;
  const uint n = elements_per_block;

  // block-wise scan, save "block sum" into increments
  uint offset = 1;
  uint ai = thread;
  uint bi = thread + (n / 2);
  uint bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  uint bankOffsetB = CONFLICT_FREE_OFFSET(bi);
  uint ai_global = ai + global_offset;
  uint bi_global = bi + global_offset;
  tmp[ai + bankOffsetA] = (ai_global < len) ? increments[ai_global] : 0;
  tmp[bi + bankOffsetB] = (bi_global < len) ? increments[bi_global] : 0;
  for (uint d = n >> 1; d > 0; d >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thread < d) {
      uint ai = offset * (2 * thread + 1) - 1;
      uint bi = offset * (2 * thread + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);
      tmp[bi] += tmp[ai];
    }
    offset *= 2;
  }
  if (thread == 0)
    tmp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
  for (uint d = 1; d < n; d *= 2) {
    offset >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thread < d) {
      int ai = offset * (2 * thread + 1) - 1;
      int bi = offset * (2 * thread + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);
      uint t = tmp[ai];
      tmp[ai] = tmp[bi];
      tmp[bi] += t;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (ai_global < len) increments[ai_global] = tmp[ai + bankOffsetA];
  if (bi_global < len) increments[bi_global] = tmp[bi + bankOffsetB];
}

kernel void es_phase_3(global uint* restrict data,
                       global uint* restrict increments,
                       uint len) {
  const uint thread = get_local_id(0);
  const uint block = get_group_id(0);
  const uint threads_per_block = get_local_size(0);
  const uint elements_per_block = threads_per_block * 2;
  const uint global_offset = block * elements_per_block;
  const uint n = elements_per_block;
  const uint index = global_offset + thread;

  // add the appropriate value to each block
  uint ai = 2 * thread;
  uint bi = 2 * thread + 1;
  uint ai_global = ai + global_offset;
  uint bi_global = bi + global_offset;
  uint increment = increments[block];
  if (ai_global < len) data[ai_global] += increment;
  if (bi_global < len) data[bi_global] += increment;
}
