/******************************************************************************
 * function: produce_chunk_id_literals                                       *
 * input:                                                                     *
 * output:                                                                    *
 ******************************************************************************/


uint mod31c(uint x);
uint mod31quik(uint xx);

constant private uint mask1 = (31U << 0) | (31U << 10) | (31U << 20) | (31U << 30);
constant private uint mask2 = (31U << 5) | (31U << 15) | (31U << 25);
constant private uchar t[31 * 2 /* 36 */] = {
   0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
  20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
  30,  0,  1,  2,  3,  4,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0
};

//
// Algorithms
//

kernel void produce_chunks(global uint* rids, global uint* cids,
                           global uint* lits) {
  uint thread = get_global_id(0);
  //lits[thread] = 1u << mod31c(rids[thread]);
  lits[thread] = 1u << mod31quik(rids[thread]);
  lits[thread] |= 1u << 31u;
  cids[thread] = (uint) rids[thread] / 31u;
}

kernel void produce_chunks2(global uint* rids, global uint* cids,
                            global uint* lits, uint len) {
  const uint thread = get_local_id(0);
  const uint block = get_group_id(0);
  const uint threads_per_block = get_local_size(0);
  const uint elements_per_block = threads_per_block;
  const uint offset = block * elements_per_block;
  const uint n = min(elements_per_block, len - offset);

  event_t copies[3];
  local uint l_rids[1024];
  local uint l_cids[1024];
  local uint l_lits[1024];

  copies[0] = async_work_group_copy(l_rids, rids + offset, n, 0);
  copies[1] = async_work_group_copy(l_cids, cids + offset, n, 0);
  copies[2] = async_work_group_copy(l_lits, lits + offset, n, 0);
  wait_group_events(3, copies);
  barrier(CLK_LOCAL_MEM_FENCE);

  if (thread < n) {
    l_lits[thread]  = 1u << mod31c(l_rids[thread]);
    l_lits[thread] |= 1u << 31u;
    l_cids[thread]  = l_rids[thread] / 31u;
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  copies[0] = async_work_group_copy(cids + offset, l_cids, n, 0);
  copies[1] = async_work_group_copy(lits + offset, l_lits, n, 0);
  wait_group_events(2, copies);
}

kernel void produce_chunks3(global uint* rids, global uint* cids,
                            global uint* lits, uint len) {
  const uint thread = get_local_id(0);
  const uint block = get_group_id(0);
  const uint threads_per_block = get_local_size(0);
  const uint elements_per_block = threads_per_block * 2;
  const uint offset = block * elements_per_block;
  const uint n = min(elements_per_block, len - offset);

  event_t copies[3];
  local uint l_rids[1024 * 2];
  local uint l_cids[1024 * 2];
  local uint l_lits[1024 * 2];

  copies[0] = async_work_group_copy(l_rids, rids + offset, n, 0);
  copies[1] = async_work_group_copy(l_cids, cids + offset, n, 0);
  copies[2] = async_work_group_copy(l_lits, lits + offset, n, 0);
  wait_group_events(3, copies);
  barrier(CLK_LOCAL_MEM_FENCE);

  uint ai = 2 * thread;
  uint bi = 2 * thread + 1;
  if (ai < n) {
    uint tmp = l_rids[ai];
    l_lits[ai]  = 1u << mod31c(tmp);
    l_lits[ai] |= 1u << 31u;
    l_cids[ai]  = tmp / 31u;
  }
  if (bi < n) {
    uint tmp = l_rids[bi];
    l_lits[bi]  = 1u << mod31c(tmp);
    l_lits[bi] |= 1u << 31u;
    l_cids[bi]  = tmp / 31u;
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  copies[0] = async_work_group_copy(cids + offset, l_cids, n, 0);
  copies[1] = async_work_group_copy(lits + offset, l_lits, n, 0);
  wait_group_events(2, copies);
}

//
// Utility functions
//

// fast mod 31 implementations
// https://stackoverflow.com/questions/26047196/is-there-any-way-to-write-mod-31-without-modulus-division-operators
uint mod31c(uint x) {
  x = (x & mask1) + ((x & mask2) >> 5);
  x += x >> 20;
  x += x >> 10;

  x = (x & 31) + ((x >> 5) & 31);
  return x >= 31 ? x - 31: x;
}

uint mod31quik(uint xx) {
  #define mask (31u | (31u << 10) | (31u << 20) | (31u << 30))
  uint x = (xx & mask) + ((xx >> 5) & mask);
  x += x >> 20;
  x += x >> 10;
  x = (x & 31u) + ((x >> 5) & 31u);
  return t[x];
}

