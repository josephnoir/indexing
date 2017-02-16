/******************************************************************************
 * function: merge_lit_by_val_chids                                          *
 * input:                                                                     *
 * output:                                                                    *
 ******************************************************************************/

/**
 * High level function:
 * size_t merge_lit_by_val_chids(global uint* input, global uint* chids,
 *                                global uint* lits, size_t li, size_t work_size); 
 */


// prototypes
uint sumReduce128(local uint* arr);
int compactSIMDPrefixSum(local const uint* dsData, local const uint* dsValid,
                         local uint* dsCompact,    local uint* dsLocalIndex);
int exclusivePrescan128(local const uint* in, local uint* outAndTemp);

/**
 * Steps
 * - create heads
 * - lazy_segmented_scan
 * - stream compaction to reduce all that have a 0 in heads?
 * (There is probably a more efficient solution)
 */

kernel void create_heads(global uint* keys_high, global uint* keys_low,
                         global uint* heads) {
  uint idx = get_global_id(0);
  heads[idx] = 0;
  if (idx > 0) {
    if ((keys_high[idx] != keys_high[idx - 1]) ||
        (keys_low [idx] != keys_low [idx - 1]))
    heads[idx] = 1;
  } else {
    heads[idx] = 1;
  }
}

// somewhat inefficient segmented scan (not really a scan as it stores the
// accumulated value at the beginning of the segment)
kernel void lazy_segmented_scan(global uint* heads, global uint* data) {
  uint idx = get_global_id(0);
  uint maximum = get_global_size(0);
  if (heads[idx] != 0) {
    uint val = data[idx];
    uint curr = idx + 1;
    while (heads[curr] == 0 && curr < maximum) {
      val |= data[curr]; // OR operation
      curr += 1;
    }
    data[idx] = val;
  }
}

/*
// exclusive scan on ...
kernel void exclusive_scan() {

}

// compact to keep only the 
kernel void stream_compaction(global uint* config, global uint* heads,
                              global uint* keys_low, global uint* heads,
                              global uint* values) {

}
*/

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

// Phase 1: Count valid elements per thread block
// Hard-code 128 thd/blk
uint sumReduce128(local uint* arr) {
    // Parallel reduce element counts
    // Assumes 128 thd/block
    int idx = get_local_id(0);
    if (idx < 64) arr[idx] += arr[idx+64];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (idx < 32) arr[idx] += arr[idx+32];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (idx < 16) arr[idx] += arr[idx+16];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (idx < 8) arr[idx] += arr[idx+8];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (idx < 4) arr[idx] += arr[idx+4];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (idx < 2) arr[idx] += arr[idx+2];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (idx < 1) arr[idx] += arr[idx+1];
    barrier(CLK_LOCAL_MEM_FENCE);
    return arr[0];
}

// len in config
kernel void countElts(global uint* restrict config,
                      global uint* restrict dgBlockCounts,
                      global uint* restrict dgValid,
                      local  uint* restrict dsCount) {
  uint len = config[0];
  uint idx  = get_local_id(0);
  uint gidx = get_group_id(0);
  uint ngrps = get_num_groups(0);
  uint lsize = get_local_size(0);
  dsCount[idx] = 0;
  // epb: eltsPerBlock
  const uint epb = len / get_num_groups(0) + ((len % ngrps) ? 1 : 0);
  uint ub
    = (len < (gidx + 1) * epb) ? len : ((gidx + 1) * epb);
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
int exclusivePrescan128(local const uint* in, local uint* outAndTemp) {
  uint idx = get_local_id(0);
  const uint n = 128;
  //TODO: this temp storage could be reduced since we write to shared 
  //      memory in out anyway, and n is hardcoded
  //__shared__ int temp[2*n];
  local uint* temp = outAndTemp;
  int pout = 1;
  int pin  = 0;

  // load input into temp
  // This is exclusive scan, so shift right by one and set first elt to 0
  temp[pout * n + idx] = (idx > 0) ? in[idx - 1] : 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint offset = 1; offset < n; offset *= 2) {
    pout = 1 - pout; // swap double buffer indices
    pin  = 1 - pout;
    barrier(CLK_LOCAL_MEM_FENCE);
    temp[pout * n + idx] = temp[pin * n + idx];
    if (idx >= offset)
      temp[pout * n + idx] += temp[pin * n + idx - offset];
  }
  //out[idx] = temp[pout*n+idx]; // write output
  barrier(CLK_LOCAL_MEM_FENCE);
  return outAndTemp[127] + in[127]; // Return sum of all elements
}

int compactSIMDPrefixSum(local const uint* dsData, local const uint* dsValid,
                         local uint* dsCompact,    local uint* dsLocalIndex) {
  uint idx = get_local_id(0);
  int numValid = exclusivePrescan128(dsValid,dsLocalIndex);
  if (dsValid[idx])
    dsCompact[dsLocalIndex[idx]] = dsData[idx];
  return numValid;
}

kernel void moveValidElementsStaged(global       uint* restrict config,
                                    global const uint* restrict dgData,
                                    global       uint* restrict dgCompact,
                                    global const uint* restrict dgValid,
                                    global const uint* restrict dgBlockCounts,
                                    local        uint* restrict inBlock,
                                    local        uint* restrict validBlock,
                                    local        uint* restrict compactBlock) {
  uint len = config[0];
  // dNumValidElements = config[1]
  uint idx = get_local_id(0);
  uint gidx = get_group_id(0);
  uint ngrps = get_num_groups(0);
  uint lsize = get_local_size(0);
  local uint dsLocalIndex[256];
  int blockOutOffset=0;
  // Sum up the blockCounts before us to find our offset
  // This is totally inefficient - lots of repeated work b/w blocks,
  // and uneven balancing. Paper implements this as a prefix sum kernel
  // in phase II. May still be faster than an extra kernel invocation?
  for (uint base = 0; base < gidx; base += lsize) {
    // Load up the count of valid elements for each block before us in batches of 128
    validBlock[idx] = 0;
    if ((base + idx) < gidx)
        validBlock[idx] = dgBlockCounts[base + idx];
    barrier(CLK_LOCAL_MEM_FENCE);
    // Parallel reduce these counts
    // Accumulate in the final offset variable
    blockOutOffset += sumReduce128(validBlock);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  // epb = eltsPerBlock
  const uint epb = len / ngrps + ((len % ngrps) ? 1 : 0);
  uint ub = (len < (gidx + 1) * epb) ? len : ((gidx + 1) * epb);
  for (uint base = gidx * epb; base < (gidx + 1) * epb; base += lsize) {
    validBlock[idx] = 0;
    if ((base + idx) < ub) {
      validBlock[idx] = dgValid[base + idx];
      inBlock[idx] = dgData[base + idx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint numValidBlock
      = compactSIMDPrefixSum(inBlock, validBlock, compactBlock, dsLocalIndex);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (idx < numValidBlock) {
      dgCompact[blockOutOffset + idx] = compactBlock[idx];
    }
    blockOutOffset += numValidBlock;
  }
  if (gidx == (ngrps - 1) && idx == 0)
    config[1] = blockOutOffset;
}
