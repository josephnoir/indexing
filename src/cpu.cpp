
#include <cmath>
#include <vector>
#include <cassert>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <iostream>
#include <algorithm>

using namespace std;

namespace {

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

template<class Key, class Lit>
size_t reduce_by_key(vector<Key>& keys, vector<Lit>& lits) {
  vector<Key> new_keys;
  vector<Lit> new_lits;
  size_t from = 0;
  while (from < keys.size()) {
    new_keys.push_back(keys[from]);
    auto to = from;
    while (to < keys.size() && keys[to] == keys[from])
      ++to;
    /*
    cout << "Bounds: " << from << " - " << to << endl;
    for (size_t i = from; i < to; ++i) {
      cout << as_binary(keys[i]) << " --> " << as_binary(lits[i]) << endl;
    }
    */
    Lit merged_lit = 0;
    while (from < to) {
      merged_lit |= lits[from];
      ++from;
    }
    new_lits.push_back(merged_lit);
  }
  lits.clear();
  keys.clear();
  lits = move(new_lits);
  keys = move(new_keys);
  assert(keys.size() == lits.size());
  return lits.size();
}

template<class T>
size_t reduce_by_key(vector<T>& values, vector<T>& chids, vector<T>& lits) {
  assert(values.size() == chids.size());
  assert(chids.size() == lits.size());
  auto max = values.size();
  vector<T> new_values;
  vector<T> new_chids;
  vector<T> new_lits;
  size_t from = 0;
  while (from < max) {
    new_values.push_back(values[from]);
    new_chids.push_back(chids[from]);
    auto to = from;
    while (to < max && values[to] == values[from] && chids[to] == chids[from])
      ++to;
    /*
    cout << "Bounds: " << from << " - " << to << endl;
    for (size_t i = from; i < to; ++i) {
      cout << as_binary(values[i]) << as_binary(chids[i])
           << " --> " << as_binary(lits[i]) << endl;
    }
    */
    T merged_lit = 0;
    while (from < to) {
      merged_lit |= lits[from];
      ++from;
    }
    new_lits.push_back(merged_lit);
  }
  values.clear();
  chids.clear();
  lits.clear();
  values = move(new_values);
  chids = move(new_chids);
  lits = move(new_lits);
  assert(values.size() == chids.size());
  assert(chids.size() == lits.size());
  return lits.size();
}

template<class T>
size_t stream_compaction(vector<T>& index, T val = 0) {
  index.erase(remove(begin(index), end(index), val), end(index));
  return index.size();
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
  // auto prev_size = values.size();
  auto k = reduce_by_key(values, chids, lits);
  /*
  cout << "Reduced from " << prev_size << " to " << k << " values" << endl;
  for (size_t i = 0; i < k; ++i) {
    cout << as_binary(values[i]) << as_binary(chids[i])
         << " --> " << as_binary(lits[i]) << endl;
  }
  */
  /*
  vector<uint64_t> keys(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    keys[i] = (uint64_t(values[i]) << 32) | chids[i];
  }
  // algorithm: reduce by key
  for (size_t i = 0; i < values.size(); ++i) {
    cout << as_binary(keys[i]) << " --> " << as_binary(lits[i]) << endl;
  }
  auto prev_size = keys.size();
  auto k = reduce_by_key(keys, lits);
  cout << "Reduced from " << prev_size << " to " << k << " values" << endl;
  for (size_t i = 0; i < k; ++i) {
    cout << as_binary(keys[i]) << " --> " << as_binary(lits[i]) << endl;
  }
  // move keys back to values and chids
  for (size_t i = 0; i < keys.size(); ++i) {
    // keys[i] = (uint64_t(values[i]) << 32) | chids[i];
    values[i] = static_cast<uint32_t>(keys[i] >> 32);
    chids[i]  = static_cast<uint32_t>(keys[i]);
  }
  */


  // ### produce fills ###
  vector<uint64_t> heads(k);
  adjacent_difference(begin(values), end(values), begin(heads));
  heads.front() = 1;
  //for (size_t i = 0; i < values.size(); ++i) {
  //  cout << as_binary(keys[i]) << " --> " << heads[i] << endl;
  //}
  for (size_t i = 0; i < k; ++i) {
    if (heads[i] == 0) {
      chids[i] = chids[i] - chids[i] - 1;
    } else {
      chids[i] = chids[i] - 1;
    }
  }


  // ### fuse fill literals ###
  vector<uint32_t> index(2 * k);
  for (size_t i = 0; i < k; ++i) {
    index[2 * i] = chids[i];
    index[2 * i + 1] = lits[i];
  }
  auto idx_length = stream_compaction(index);
  cout << "Created index of length: " << idx_length << endl;


  // ### compute comlumn len ###

}

