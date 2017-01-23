
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

// Reduce by key for OR operation
/*
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
    cout << "Bounds: " << from << " - " << to << endl;
    for (size_t i = from; i < to; ++i) {
      cout << as_binary(keys[i]) << " --> " << as_binary(lits[i]) << endl;
    }
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
*/

template<class T>
size_t reduce_by_key(vector<T>& input, vector<T>& chids, vector<T>& lits) {
  assert(input.size() == chids.size());
  assert(chids.size() == lits.size());
  auto max = input.size();
  vector<T> new_input;
  vector<T> new_chids;
  vector<T> new_lits;
  size_t from = 0;
  while (from < max) {
    new_input.push_back(input[from]);
    new_chids.push_back(chids[from]);
    auto to = from;
    while (to < max && input[to] == input[from] && chids[to] == chids[from])
      ++to;
    /*
    cout << "Bounds: " << from << " - " << to << endl;
    for (size_t i = from; i < to; ++i) {
      cout << as_binary(input[i]) << as_binary(chids[i])
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
  input.clear();
  chids.clear();
  lits.clear();
  input = move(new_input);
  chids = move(new_chids);
  lits = move(new_lits);
  assert(input.size() == chids.size());
  assert(chids.size() == lits.size());
  return lits.size();
}

// Reduce by key for sum operation
template<class T>
size_t reduce_by_key(vector<T>& keys, vector<T>& vals) {
  vector<T> new_keys;
  vector<T> new_vals;
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
    T merged_lit = 0;
    while (from < to) {
      merged_lit += vals[from];
      ++from;
    }
    new_vals.push_back(merged_lit);
  }
  vals.clear();
  keys.clear();
  vals = move(new_vals);
  keys = move(new_keys);
  assert(keys.size() == lits.size());
  return vals.size();
}

template<class T>
vector<T> inclusive_scan(const vector<T>& vals) {
  vector<T> results(vals.size());
  results[0] = vals[0];
  for (size_t i = 1; i < vals.size(); ++i) {
    results[i] = results[i - 1] + vals[i];
  }
  return results;
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

  auto input = values;

  // ### sort rids by value ###
  vector<uint32_t> rids(input.size());
  iota(begin(rids), end(rids), 0);
  for (size_t i = (input.size() - 1); i > 0; --i) {
    for (size_t j = 0; j < i; ++j) {
      if (input[j] > input[j + 1]) {
        // switch input
        auto tmp = input[j];
        input[j] = input[j + 1];
        input[j + 1] = tmp;
        // switch rids
        tmp = rids[j];
        rids[j] = rids[j + 1];
        rids[j + 1] = tmp;
      }
    }
  }
  // for (size_t i = 0; i < input.size(); ++i) {
  //   cout << input[i] << " --> " << rids[i] << endl;
  // }


  // ### produce chuck id literal ###
  vector<uint32_t> chids(input.size());
  vector<uint32_t> lits(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    chids[i] = rids[i] / 31;
    lits[i]  = 0x1 << (rids[i] % 31);
    lits[i] |= 0x1 << 31;
  }


  // ### merge lit by val chids ###
  // auto prev_size = input.size();
  auto k = reduce_by_key(input, chids, lits);
  /*
  cout << "Reduced from " << prev_size << " to " << k << " input" << endl;
  for (size_t i = 0; i < k; ++i) {
    cout << as_binary(input[i]) << as_binary(chids[i])
         << " --> " << as_binary(lits[i]) << endl;
  }
  */
  /*
  vector<uint64_t> keys(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    keys[i] = (uint64_t(input[i]) << 32) | chids[i];
  }
  // algorithm: reduce by key
  for (size_t i = 0; i < input.size(); ++i) {
    cout << as_binary(keys[i]) << " --> " << as_binary(lits[i]) << endl;
  }
  auto prev_size = keys.size();
  auto k = reduce_by_key(keys, lits);
  cout << "Reduced from " << prev_size << " to " << k << " input" << endl;
  for (size_t i = 0; i < k; ++i) {
    cout << as_binary(keys[i]) << " --> " << as_binary(lits[i]) << endl;
  }
  // move keys back to input and chids
  for (size_t i = 0; i < keys.size(); ++i) {
    // keys[i] = (uint64_t(input[i]) << 32) | chids[i];
    input[i] = static_cast<uint32_t>(keys[i] >> 32);
    chids[i]  = static_cast<uint32_t>(keys[i]);
  }
  */


  // ### produce fills ###
  // in : input, chids, k
  // out: chids
  vector<uint64_t> heads(k);
  adjacent_difference(begin(input), end(input), begin(heads));
  heads.front() = 1; // not sure about this one
  //for (size_t i = 0; i < input.size(); ++i) {
  //  cout << as_binary(keys[i]) << " --> " << heads[i] << endl;
  //}
  for (size_t i = 0; i < k; ++i) {
    if (heads[i] == 0) {
      chids[i] = chids[i] - chids[i - 1] - 1;
    } else {
      if (chids[i] != 0)
        chids[i] = chids[i] - 1;
    }
  }


  // ### fuse fill literals ###
  // in : chids, lits, k
  // out: index, index_length
  vector<uint32_t> index(2 * k);
  for (size_t i = 0; i < k; ++i) {
    index[2 * i] = chids[i];
    index[2 * i + 1] = lits[i];
  }
  auto idx_length = stream_compaction(index, 0u);
  cout << "Created index of length: " << idx_length << endl;


  // ### compute comlumn len ###
  // in : chids, input, n
  // out: keycnt, offsets
  vector<uint32_t> tmp(k);
  for (size_t i = 0; i < k; ++i) {
    tmp[i] = (1 + (chids[i] == 0 ? 0 : 1));
  }
  auto keycnt = reduce_by_key(input, tmp);
  cout << "Created index for values:" << endl;
  for (size_t i = 0; i < values.size(); ++i) {
    cout << "> " << values[i] << endl;
  }
  vector<uint32_t> offsets = inclusive_scan(tmp);


  // Searching for some sensible output format
  for (size_t i = 0; i < keycnt; ++i) {
    auto value = input[i];
    auto length = tmp[i];
    auto offset = offsets[i];
    cout << "Index for value " << value << ":" << endl
         << "> has length " << length << " and offset " << offset << endl;
    //for (size_t j = 0; j < length; ++j) {
    //  cout << "[" << j << "] " << as_binary(index[offset + j]) << endl;
    //}
    //cout << endl;
  }
  cout << "Complete index is:" << endl;
  for (auto& i : index) {
    cout << "> " << as_binary(i) << endl;
  }
}

