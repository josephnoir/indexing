/*
 *  Code for CUDA stream compaction. Roughly based on:
 *  Billeter M, Olsson O, Assarsson U.
 *  Efficient Stream Compaction on Wide SIMD Many-Core Architectures.
 *      High Performance Graphics 2009.
 *
 *  Project: https://github.com/pandegroup/openmm
 *  File:    platforms/opencl/src/kernels/compact.cl
 *  Notes:
 *    - paper recommends 128 threads/block, so this is hard coded.
 *    - I only implement the prefix-sum based compact primitive,
 *      and not the POPC one, as that is more complicated and performs poorly
 *      on current hardware
 *    - I only implement the scattered- and staged-write variant of phase III
 *      as it they have reasonable performance across most of the tested
 *      workloads in the paper. The selective variant is not implemented.
 *    - The prefix sum of per-block element counts (phase II) is not done in
 *      a particularly efficient manner. It is, however, done in a very easy
 *      to program manner, and integrated into the top of phase III, reducing
 *      the number of kernel invocations required. If one wanted to use
 *      existing code, it'd be easy to take the CUDA SDK scanLargeArray
 *      sample, and do a prefix sum over dgBlockCounts in a phase II kernel.
 *      You could also adapt the existing prescan128 to take an initial value,
 *      and scan dgBlockCounts in stages.
 *
 * Date:         23 Aug 2009
 * Author:       CUDA version by Imran Haque (ihaque@cs.stanford.edu),
 *               converted to OpenCL by Peter Eastman
 * Affiliation:  Stanford University
 * License:      Public Domain
 */

#define N 128

// prototypes
uint sumReduce128(local uint* arr);
uint compactSIMDPrefixSum(local const uint* dsData, local const uint* dsValid,
                          local uint* dsCompact,    local uint* dsLocalIndex);
uint exclusivePrescan128(local const uint* in, local uint* outAndTemp);


// Phase 1: Count valid elements per thread block
// Hard-code 128 thd/blk
uint sumReduce128(local uint* arr) {
  // Parallel reduce element counts, assumes 128 thd/block
  const uint idx = get_local_id(0);
  if (idx < 64) arr[idx] += arr[idx + 64];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (idx < 32) arr[idx] += arr[idx + 32];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (idx < 16) arr[idx] += arr[idx + 16];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (idx <  8) arr[idx] += arr[idx +  8];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (idx <  4) arr[idx] += arr[idx +  4];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (idx <  2) arr[idx] += arr[idx +  2];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (idx <  1) arr[idx] += arr[idx +  1];
  barrier(CLK_LOCAL_MEM_FENCE);
  return arr[0];
}

kernel void countElts(global uint* restrict dgBlockCounts,
                      global uint* restrict dgValid,
                      local  uint* restrict dsCount,
                      private const uint len) {
  const uint idx  = get_local_id(0);
  const uint gidx = get_group_id(0);
  const uint ngrps = get_num_groups(0);
  const uint lsize = get_local_size(0);
  const uint epb = len / ngrps + ((len % ngrps) ? 1 : 0);
  const uint ub  = (len < (gidx + 1) * epb) ? len : ((gidx + 1) * epb);
  dsCount[idx] = 0;
  for (uint base = gidx * epb; base < (gidx + 1) * epb; base += lsize) {
    if ((base + idx) < ub && dgValid[base + idx])
      dsCount[idx]++;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint blockCount = sumReduce128(dsCount);
  if (idx == 0)
      dgBlockCounts[gidx] = blockCount;
  return;
}

// Phase 2/3: Move valid elements using SIMD compaction. Phase 2 is done
// implicitly at top of global__ method. Exclusive prefix scan over 128
// elements, assumes 128 threads. Taken from cuda SDK "scan" sample for
// naive scan, with small modifications.
uint exclusivePrescan128(local const uint* in, local uint* outAndTemp) {
  const uint idx = get_local_id(0);
  int pout = 1;
  int pin  = 0;

  // load input into outAndTemp
  // This is exclusive scan, so shift right by one and set first elt to 0
  outAndTemp[pout * N + idx] = (idx > 0) ? in[idx - 1] : 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint offset = 1; offset < N; offset *= 2) {
    pout = 1 - pout; // swap double buffer indexes
    pin  = 1 - pout;
    barrier(CLK_LOCAL_MEM_FENCE);
    outAndTemp[pout * N + idx] = outAndTemp[pin * N + idx];
    if (idx >= offset)
      outAndTemp[pout * N + idx] += outAndTemp[pin * N + idx - offset];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  return outAndTemp[N - 1] + in[N - 1]; // Return sum of all elements
}

uint compactSIMDPrefixSum(local const uint* dsData, local const uint* dsValid,
                         local uint* dsCompact,    local uint* dsLocalIndex) {
  uint idx = get_local_id(0);
  uint numValid = exclusivePrescan128(dsValid,dsLocalIndex);
  if (dsValid[idx])
    dsCompact[dsLocalIndex[idx]] = dsData[idx];
  return numValid;
}

// dgValid and dgData may be the same data if we want to keep everything != 0
kernel void moveValidElementsStaged(global       uint* restrict result,
                                    global const uint*          dgData,
                                    global       uint* restrict dgCompact,
                                    global const uint*          dgValid,
                                    global const uint* restrict dgBlockCounts,
                                    local        uint* restrict inBlock,
                                    local        uint* restrict validBlock,
                                    local        uint* restrict compactBlock,
                                    private const uint          len) {
  uint idx = get_local_id(0);
  uint gidx = get_group_id(0);
  uint ngrps = get_num_groups(0);
  uint lsize = get_local_size(0);
  local uint dsLocalIndex[256];
  uint blockOutOffset = 0;
  // for (uint base = 0; base < gidx; base += lsize) {
  //   // Load up the count of valid elements for each block before us
  //   // in batches of 128
  //   if ((base + idx) < gidx)
  //     validBlock[idx] = dgBlockCounts[base + idx];
  //   else
  //     validBlock[idx] = 0;
  //   barrier(CLK_LOCAL_MEM_FENCE);
  //   // Parallel reduce these counts, accumulate in the final offset variable
  //   blockOutOffset += sumReduce128(validBlock);
  //   barrier(CLK_LOCAL_MEM_FENCE);
  // }
  blockOutOffset = dgBlockCounts[gidx];
  const uint epb = len / ngrps + ((len % ngrps) ? 1 : 0);
  const uint ub = (len < (gidx + 1) * epb) ? len : ((gidx + 1) * epb);
  for (uint base = gidx * epb; base < (gidx + 1) * epb; base += lsize) {
    if ((base + idx) < ub) {
      validBlock[idx] = (dgValid[base + idx] != 0);
      inBlock[idx] = dgData[base + idx];
    } else {
      validBlock[idx] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint numValidBlock = compactSIMDPrefixSum(inBlock, validBlock,
                                              compactBlock, dsLocalIndex);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (idx < numValidBlock)
      dgCompact[blockOutOffset + idx] = compactBlock[idx];
    blockOutOffset += numValidBlock;
  }
  if (gidx == (ngrps - 1) && idx == 0)
    result[0] = blockOutOffset;
}
