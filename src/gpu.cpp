
#include <cmath>
#include <random>
#include <vector>
#include <cassert>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <iostream>
#include <algorithm>

#include "caf/all.hpp"
#include "caf/opencl/all.hpp"

using namespace std;
using namespace caf;
using namespace caf::opencl;

namespace {

constexpr const char* kernel_name = "kernel_wah_index";
constexpr const char* kernel_file = "kernel_wah_index.cl";
constexpr const char* kernel_source = R"__(
void sort_rids_by_value(__global uint* input, __global  uint* rids,
                        size_t idx, size_t total);
void parallel_selection_sort(__global uint* key, __global uint* data,
                             size_t idx, size_t total);
void produce_chunck_id_literals(__global uint* rids, __global uint* chids,
                                __global uint* lits, size_t idx, size_t total);

__kernel void kernel_wah_index(__global uint* input,
                               __global uint* config,
                               __global uint* index,
                               __global uint* offsets,
                               __global uint* rids,
                               __global uint* chids,
                               __global uint* lits) {
  // just 1 dimension here: 0
  size_t total = get_global_size(0);
  size_t idx = get_global_id(0);
  // read config data
  // assumed to be == total, except idx, which is == 2 x total
  //uint input_size   = config[0];
  //uint rids_size    = config[1];
  //uint chids_size   = config[2];
  //uint lits_size    = config[3];
  //uint index_size   = config[4];
  //uint offsets_size = config[5];

  //__gloabl uint[index_size] rids;
  sort_rids_by_value(input, rids, idx, total);
  barrier(CLK_GLOBAL_MEM_FENCE);
  //produce_chunck_id_literals(rids, chids, lits, idx, total);
  //barrier(CLK_GLOBAL_MEM_FENCE);
/*
  auto k = merged_lit_by_val_chids(input, chids, lits);
  produce_fills(input, chids, k);
  vector<uint> index(2 * k);
  auto idx_length = fuse_fill_literals(chids, lits, index, k);
  vector<uint> offsets(k);
  auto keycnt = compute_colum_length(input, chids, offsets, k);
*/
}

void sort_rids_by_value(__global uint* input, __global uint* rids,
                        size_t idx, size_t total) {
  rids[idx] = idx;
  // sort by input value
  parallel_selection_sort(input, rids, idx, total);
}

void produce_chunck_id_literals(__global uint* rids, __global uint* chids,
                                __global uint* lits, size_t idx, size_t total) {
  (void) total;
  lits[idx] = rids[idx] % 31u;
  lits[idx] |= 1u << 31;
  chids[idx] = (uint) rids[idx] / 31;
}

/**
 * Sort found on: http://www.bealto.com/gpu-sorting_intro.html
 * License:set syntax=:
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

// One thread per record
void parallel_selection_sort(__global uint* key, __global uint* data,
                             size_t idx, size_t total) {
  uint key_value = key[idx];
  uint data_value = data[idx];
  // Compute position of in[i] in output
  int pos = 0;
  for (size_t j = 0; j < total; ++j) {
    uint curr = key[j]; // broadcasted
    bool smaller = (curr < key_value) || (curr == key_value && j < idx);
    pos += (smaller) ? 1 : 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  key[pos] = key_value;
  data[pos] = data_value;
}
)__";


template<class T>
string as_binary(T num) {
  stringstream s;
  auto num_bits = (sizeof(T) * 8);
  T mask = T(0x1) << (num_bits - 1);
  while (mask > 0) {
    s << ((num & mask) ? "1" : "0");
    mask >>= 1;
  }
  return s.str();
}

void indexer(event_based_actor* self) {
  size_t amount = 10;
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_int_distribution<uint32_t> dist(0,10);
  vector<uint32_t> values(amount);
  for (size_t i = 0; i < amount; ++i)
    values[i] = dist(rng);

/*
  for (auto& val : values)
    cout << val << " ";
  cout << endl;
*/

  vector<uint32_t> input = values;
  vector<uint32_t> index(2 * amount);
  vector<uint32_t> offsets(amount);

  vector<uint32_t> config{static_cast<uint32_t>(amount),      // input
                          static_cast<uint32_t>(amount),      // rids
                          static_cast<uint32_t>(amount),      // chids
                          static_cast<uint32_t>(amount),      // lits
                          static_cast<uint32_t>(amount * 2),  // index
                          static_cast<uint32_t>(amount)};     // offsets

  auto worker = self->system().opencl_manager().spawn(
    kernel_source, kernel_name,
    spawn_config{dim_vec{amount}},
    in<vector<uint32_t>>{},             // input
    in_out<vector<uint32_t>>{},         // config
    out<vector<uint32_t>>{},            // index
    out<vector<uint32_t>>{},            // offsets
    buffer<vector<uint32_t>>{amount},   // rids
    buffer<vector<uint32_t>>{amount},   // chids
    buffer<vector<uint32_t>>{amount}    // lits
  );
  // send both matrices to the actor and wait for a result
  self->request(worker, chrono::seconds(30), move(input), move(config)).then(
    [](const vector<uint32_t>& config, const vector<uint32_t>& index,
       const vector<uint32_t>& offsets) {
      cout << "received some stuff!" << endl;
      static_cast<void>(index);
      static_cast<void>(offsets);
      static_cast<void>(config);

      // cout << "Created index for " << amount
      //      << " values, with " << idx_length
      //      << " entries" << endl;

      // // Searching for some sensible output format
      // for (size_t i = 0; i < keycnt; ++i) {
      //   auto value = input[i];
      //   auto offset = offsets[i];
      //   auto length = (i == keycnt - 1) ? index.size() - offsets[i]
      //                                   : offsets[i + 1] - offsets[i];
      //   cout << "Index for value " << value << ":" << endl
      //        << "> length " << length << endl << "> offset " << offset << endl;
      //   for (size_t j = 0; j < length; ++j) {
      //     cout << as_binary(index[offset + j]) << " ";
      //   }
      //   cout << endl << endl;
      // }
    }
  );
}

} // namespace <anonymous>

int main() {
  actor_system_config cfg;
  cfg.load<opencl::manager>()
     .add_message_type<vector<uint32_t>>("data_vector");
  actor_system system{cfg};
  system.spawn(indexer);
  system.await_all_actors_done();
}

