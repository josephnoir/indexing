/******************************************************************************
 * Copyright (C) 2017                                                         *
 * Raphael Hiesgen <raphael.hiesgen (at) haw-hamburg.de>                      *
 *                                                                            *
 * Distributed under the terms and conditions of the BSD 3-Clause License.    *
 *                                                                            *
 * If you did not receive a copy of the license files, see                    *
 * http://opensource.org/licenses/BSD-3-Clause and                            *
 ******************************************************************************/

// For optimizations regarding bank conflicts, look at:
// - http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html

/// Global exclusive scan, phase 1. From: 
/// - http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
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
  tmp[2 * thread] = (global_offset + (2 * thread) < len)
                  ? data[global_offset + (2 * thread)] : 0;
  tmp[2 * thread + 1] = (global_offset + (2 * thread + 1) < len)
                      ? data[global_offset + (2 * thread + 1)] : 0;
  // build sum in place up the tree
  for (uint d = n >> 1; d > 0; d >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
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
    barrier(CLK_LOCAL_MEM_FENCE);
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
  if (global_offset + (2 * thread) < len)
    data[global_offset + (2 * thread)] = tmp[2 * thread];
  if (global_offset + (2 * thread + 1) < len)
    data[global_offset + (2 * thread + 1)] = tmp[2 * thread + 1];
}

/// Global exclusive scan, phase 2.
kernel void es_phase_2(global uint* restrict data, // not used ...
                       global uint* restrict increments,
                       uint len) {
  local uint tmp[2048];
  uint thread = get_local_id(0);
  uint offset = 1;
  const uint n = 2048;
  // A (2 lines) --> load input into shared memory
  tmp[2 * thread] = (2 * thread < len) ? increments[2 * thread] : 0;
  tmp[2 * thread + 1] = (2 * thread + 1 < len) ? increments[2 * thread + 1] : 0;
  // build sum in place up the tree
  for (uint d = n >> 1; d > 0; d >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
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
    barrier(CLK_LOCAL_MEM_FENCE);
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
  if (2 * thread < len) increments[2 * thread] = tmp[2 * thread];
  if (2 * thread + 1 < len) increments[2 * thread + 1] = tmp[2 * thread + 1];
}

kernel void es_phase_3(global uint* restrict data,
                       global uint* restrict increments,
                       uint len) {
  const uint thread = get_local_id(0);
  const uint block = get_group_id(0);
  const uint threads_per_block = get_local_size(0);
  const uint elements_per_block = threads_per_block * 2;
  const uint global_offset = block * elements_per_block;

  // add the appropriate value to each block
  uint ai = 2 * thread;
  uint bi = 2 * thread + 1;
  uint ai_global = ai + global_offset;
  uint bi_global = bi + global_offset;
  uint increment = increments[block];
  if (ai_global < len) data[ai_global] += increment;
  if (bi_global < len) data[bi_global] += increment;
}
