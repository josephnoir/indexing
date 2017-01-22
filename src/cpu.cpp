
#include <cmath>
#include <vector>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <iostream>
#include <algorithm>

using namespace std;

namespace {

vector<uint32_t> reduce_by_key(const vector<uint64_t>& keys,
                               const vector<uint32_t>& lits) {
  vector<uint32_t> results;
  for (size_t from = 0; from < keys.size();) {
    auto to = from;
    while (keys[to] == keys[from] && to < keys.size())
      ++to;
    auto merged_lit = 0;
    for (auto i = from; from < to; ++from) {
      merged_lit |= lits[i];
    }
    results.push_back(merged_lit);
  }
  return results;
}

template<class T>
string as_binary(T num) {
  auto num_bits = (sizeof(T) * 8);

  stringstream s;
  T mask = T(0x1) << (num_bits - 1);
  while (mask > 0) {
    s << ((num & mask) ? "1" : "0");
    mask >>= 1;
  }
  return s.str();
}

} // namespace <anonymous>

int main() {
  vector<uint32_t> values{10,  7, 22,  6,  7,
                           1,  9, 42,  2,  5,
                          13,  3,  2,  1,  0,
                           1, 18, 18,  3, 13};
  // ### sort rids by value ###
  vector<uint32_t> rids(values.size());
  iota(begin(rids), end(rids), 0);
  for (size_t i = (values.size() - 1); i > 0; --i) {
    for (size_t j = 0; j < i; ++j) {
      if (values[j] > values[j + 1]) {
        // switch values
        auto tmp = values[j];
        values[j] = values[j + 1];
        values[j + 1] = tmp;
        // switch rids
        tmp = rids[j];
        rids[j] = rids[j + 1];
        rids[j + 1] = tmp;
      }
    }
  }
  // for (size_t i = 0; i < values.size(); ++i) {
  //   cout << values[i] << " --> " << rids[i] << endl;
  // }
  // ### produce chuck id literal ###
  vector<uint32_t> chids(values.size());
  vector<uint32_t> lits(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    chids[i] = rids[i] / 31;
    lits[i]  = 0x1 << (rids[i] % 31);
    lits[i] |= 0x1 << 31;
  }
  // ### merge lit by val chids ###
  vector<uint64_t> keys(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    keys[i] = (uint64_t(values[i]) << 32) | chids[i];
  }
  // algorithm: reduce by key
  auto k = reduce_by_key(keys, lits);
  //for (size_t i = 0; i < values.size(); ++i) {
  //  cout << as_binary(keys[i]) << " --> " << as_binary(lits[i]) << endl;
  //}
  // ### produce fills ###
  vector<uint64_t> heads(keys.size());
  adjacent_difference(begin(keys), end(keys), begin(heads));
  heads.front() = 1;
  //for (size_t i = 0; i < values.size(); ++i) {
  //  cout << as_binary(keys[i]) << " --> " << heads[i] << endl;
  //}
  for (size_t i = 0; i < heads.size(); ++i) {
    if (heads[i] == 0) {
      chids[i] = chids[i] - chids[i] - 1;
    } else {
      chids[i] = chids[i] - 1;
    }
  }
  // ### fuse fill literals ###
  for (size_t i = 0; i < keys.size(); ++i) {

  }
}

