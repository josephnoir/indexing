
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

constexpr const char* kernel_source = R"__(
  __kernel void wah_index(__global uint32_t* input,
                          __global uint32_t* index,
                          __global uint32_t* offsets,
                          __global uint32_t  input_size,
                          __global uint32_t  index_size,
                          __global uint32_t  offset_size) {
    /*
    sort_rids_by_value(input, rids);
    produce_chunck_id_literals(rids, chids, lits);
    auto k = merged_lit_by_val_chids(input, chids, lits);
    produce_fills(input, chids, k);
    vector<uint32_t> index(2 * k);
    auto idx_length = fuse_fill_literals(chids, lits, index, k);
    vector<uint32_t> offsets(k);
    auto keycnt = compute_colum_length(input, chids, offsets, k);
    */
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

} // namespace <anonymous>

int main() {
  auto amount = 100;
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_int_distribution<uint32_t> dist(0,10);
  vector<uint32_t> values(amount);
  for (int i = 0; i < amount; ++i)
    values[i] = dist(rng);

  auto input = values;
  vector<uint32_t> rids(input.size());
  vector<uint32_t> chids(input.size());
  vector<uint32_t> lits(input.size());

  vector<uint32_t> index(2 * amount);
  vector<uint32_t> offsets(amount);
  size_t idx_length = 0;
  size_t keycnt = 0;

  cout << "Created index for " << amount
       << " values, with " << idx_length
       << " entries" << endl;

  // Searching for some sensible output format
  for (size_t i = 0; i < keycnt; ++i) {
    auto value = input[i];
    auto offset = offsets[i];
    auto length = (i == keycnt - 1) ? index.size() - offsets[i]
                                    : offsets[i + 1] - offsets[i];
    cout << "Index for value " << value << ":" << endl
         << "> length " << length << endl << "> offset " << offset << endl;
    for (size_t j = 0; j < length; ++j) {
      cout << as_binary(index[offset + j]) << " ";
    }
    cout << endl << endl;
  }
}

