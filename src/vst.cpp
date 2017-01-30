
#include <cmath>
#include <random>
#include <vector>
#include <cassert>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <iostream>
#include <algorithm>

#include "vast/coder.hpp"
#include "vast/operator.hpp"
#include "vast/wah_bitmap.hpp"
#include "vast/bitmap_index.hpp"

#include "vast/concept/printable/to_string.hpp"
#include "vast/concept/printable/vast/bitmap.hpp"

using namespace std;
using namespace vast;

namespace {

template<class T>
string as_binary(T num) {
  stringstream s;
  auto num_bits = (sizeof(T) * 8);
  auto added_bits = 0;
  T mask = T(0x1) << (num_bits - 1);
  while (mask > 0) {
    if (added_bits == 32) {
      s << " ";
      added_bits = 0;
    }
    s << ((num & mask) ? "1" : "0");
    mask >>= 1;
    ++added_bits;
  }
  return s.str();
}

} // namespace <anonymous>

int main(void) {
  /*
  vector<uint32_t> values{10,  7, 22,  6,  7,
                           1,  9, 42,  2,  5,
                          13,  3,  2,  1,  0,
                           1, 18, 18,  3, 13,
                           5,  9,  0,  3,  2,
                          19,  5, 23, 22, 10,
                           6, 22};
  // auto amount = values.size();
  */

  size_t max_value = 32;
  size_t amount = 1024;
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_int_distribution<uint32_t> dist(0,max_value - 1);
  vector<uint32_t> values(amount);
  for (size_t i = 0; i < amount; ++i)
    values[i] = dist(rng);

  vector<uint32_t> distinct = values;
  std::sort(distinct.begin(), distinct.end());
  auto last = std::unique(distinct.begin(), distinct.end());
  distinct.erase(last, end(distinct));
  // for (auto val : values) {
  //   cout << val << " ";
  // }
  // cout << endl;

  bitmap_index<uint32_t, equality_coder<wah_bitmap>> bmi{max_value};
  for (auto& val : values) {
    bmi.push_back(val);
  }
  // for (auto& key : distinct) {
  //   cout << "Index for value " << key << ":" << endl;
  //   cout << to_string(bmi.lookup(vast::equal, key)) << endl;
  // }
  auto& coder = bmi.coder();
  auto& storage = coder.storage();
  for (auto& key : distinct) {
    cout << "Index for value " << key << ":" << endl;
    cout << to_string(storage[key]) << endl;
  }
  // for (auto& block : .blocks()) {
  //   cout << block << endl;
  // }
}
