/******************************************************************************
 * function: compute_colum_length                                             *
 * input:                                                                     *
 * output:                                                                    *
 ******************************************************************************/

/*
 * High level function:
 * kernel void sort_rids_by_value(global uint* config, global uint* input,
 *                                global uint* rids);
 *
 * Tasks:
 * - create vector rids (contains index)
 * - sort input and rids by input
 */

// Kernels to create rids

kernel void create_rids(global uint* config, global uint* input,
                        global uint* rids) {
  uint max_gi = config[0];
  uint gi = get_global_id(0);
  if (gi < max_gi) {
    rids[gi] = gi;
  }
}

// Kernels for sorting
/**
 * Sort found on: http://www.bealto.com/gpu-sorting_intro.html
 * License:
 *  This code is released under the following license (BSD-style).
 *  --
 *
 *  Copyright (c) 2011, Eric Bainville
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of Eric Bainville nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY ERIC BAINVILLE ''AS IS'' AND ANY
 *  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL ERIC BAINVILLE BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **/

// N threads
kernel void ParallelBitonic_A(global const uint * in, global uint * out,
                              int inc, int dir) {
  int i = get_global_id(0); // thread index
  int j = i ^ inc;          // sibling to compare

  // Load values at I and J
  uint iData = in[i];
  uint iKey    = iData;
  uint jData = in[j];
  uint jKey    = jData;

  // Compare
  bool smaller = (jKey < iKey) || ( jKey == iKey && j < i );
  bool swap    = smaller ^ (j < i) ^ ((dir & i) != 0);

  // Store
  out[i] = (swap) ? jData : iData;
}

// N/2 threads
kernel void ParallelBitonic_B_test(global const uint * in, global uint * out,
                                   int inc, int dir) {
  int t   = get_global_id(0); // thread index
  int low = t & (inc - 1);    // low order bits (below INC)
  int i   = (t<<1) - low;     // insert 0 at position INC
  int j   = i | inc;          // insert 1 at position INC

  // Load values at I and J
  uint iData = in[i];
  uint iKey    = iData;
  uint jData = in[j];
  uint jKey    = jData;

  // Compare
  bool smaller = (jKey < iKey) || ( jKey == iKey && j < i );
  bool swap    = smaller ^ ((dir & i) != 0);

  // Store
  out[i] = (swap) ? jData : iData;
  out[j] = (swap) ? iData : jData;
}

#define ORDER(a,b) {                                                          \
  bool swap = reverse ^ (a < b);                                              \
  uint auxa = a;                                                            \
  uint auxb = b;                                                            \
  a = (swap) ? auxb : auxa;                                                   \
  b = (swap) ? auxa : auxb;                                                   \
}

// N/2 threads
/*
kernel void ParallelBitonic_B2(global uint * data, int inc, int dir) {
  int t        = get_global_id(0); // thread index
  int low      = t & (inc - 1); // low order bits (below INC)
  int i        = (t << 1) - low; // insert 0 at position INC
  bool reverse = ((dir & i) == 0); // asc/desc order
  data        += i; // translate to first value

  // Load
  uint x0 = data[  0];
  uint x1 = data[inc];

  // Sort
  ORDER(x0, x1)

  // Store
  data[0  ] = x0;
  data[inc] = x1;
}
*/
kernel void ParallelBitonic_B2(global uint* config, global uint * data) {
  uint inc     = config[0];        // inc phase parameter
  uint dir     = config[1];        // sort oder 
  int t        = get_global_id(0); // thread index
  int low      = t & (inc - 1);    // low order bits (below INC)
  int i        = (t << 1) - low;   // insert 0 at position INC
  bool reverse = ((dir & i) == 0); // asc/desc order
  data        += i;                // translate to first value

  // Load
  uint x0 = data[  0];
  uint x1 = data[inc];

  // Sort
  ORDER(x0, x1)

  // Store
  data[0  ] = x0;
  data[inc] = x1;
}

// N/4 threads
kernel void ParallelBitonic_B4(global uint * data, int inc, int dir) {
  inc        >>= 1;
  int t        = get_global_id(0); // thread index
  int low      = t & (inc - 1); // low order bits (below INC)
  int i        = ((t - low) << 2) + low; // insert 00 at position INC
  bool reverse = ((dir & i) == 0); // asc/desc order
  data        += i; // translate to first value

  // Load
  uint x0 = data[    0];
  uint x1 = data[  inc];
  uint x2 = data[2*inc];
  uint x3 = data[3*inc];

  // Sort
  ORDER(x0, x2)
  ORDER(x1, x3)
  ORDER(x0, x1)
  ORDER(x2, x3)

  // Store
  data[    0] = x0;
  data[  inc] = x1;
  data[2*inc] = x2;
  data[3*inc] = x3;
}

#define ORDERV(x,a,b) {                                                       \
  bool swap = reverse ^ (x[a] < x[b]);                                        \
  uint auxa = x[a];                                                           \
  uint auxb = x[b];                                                           \
  x[a] = (swap) ? auxb : auxa;                                                \
  x[b] = (swap) ? auxa : auxb;                                                \
}
#define B2V(x,a) { ORDERV(x, a, a + 1) }
#define B4V(x,a) {                                                            \
  for (int i4 = 0; i4 < 2; i4++) {                                            \
    ORDERV(x, a + i4, a + i4 + 2)                                             \
  }                                                                           \
  B2V(x,a) B2V(x, a + 2)                                                      \
}
#define B8V(x,a) {                                                            \
  for (int i8 = 0; i8 < 4; i8++) {                                            \
    ORDERV(x, a + i8, a + i8 + 4)                                             \
  }                                                                           \
  B4V(x, a) B4V(x, a + 4)                                                     \
}
#define B16V(x,a) {                                                           \
  for (int i16 = 0; i16 < 8; i16++) {                                         \
    ORDERV(x, a + i16, a + i16 + 8)                                           \
  }                                                                           \
  B8V(x, a)                                                                   \
  B8V(x, a + 8)                                                               \
}

// N/8 threads
kernel void ParallelBitonic_B8(global uint * data, int inc, int dir) {
  inc        >>= 2;
  int t        = get_global_id(0);       // thread index
  int low      = t & (inc - 1);          // low order bits (below INC)
  int i        = ((t - low) << 3) + low; // insert 000 at position INC
  bool reverse = ((dir & i) == 0);       // asc/desc order
  data        += i;                      // translate to first value

  // Load
  uint x[8];
  for (int k = 0; k < 8; k++)
    x[k] = data[k * inc];

  // Sort
  B8V(x,0)

  // Store
  for (int k = 0; k < 8; k++)
    data[k * inc] = x[k];
}

// N/16 threads
kernel void ParallelBitonic_B16(global uint * data, int inc, int dir) {
  inc        >>= 3;
  int t        = get_global_id(0);       // thread index
  int low      = t & (inc - 1);          // low order bits (below INC)
  int i        = ((t - low) << 4) + low; // insert 0000 at position INC
  bool reverse = ((dir & i) == 0);       // asc/desc order
  data        += i;                      // translate to first value

  // Load
  uint x[16];
  for (int k = 0; k < 16; k++)
    x[k] = data[k * inc];

  // Sort
  B16V(x,0)

  // Store
  for (int k = 0; k < 16; k++)
    data[k * inc] = x[k];
}
