
#include <cmath>
#include <random>
#include <vector>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <iostream>
#include <algorithm>

#include "caf/all.hpp"

using namespace std;
using namespace caf;

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

string decoded_bitmap(const vector<uint32_t>& bitmap) {
  //cout << endl;
  //for (auto t : bitmap) {
  //  cout << as_binary(t) << " ";
  //}
  //cout << endl;
  stringstream s;
  for (auto& block : bitmap) {
    if (block & (0x1 << 31)) {
      uint32_t mask = 0x1;
      for (int i = 0; i < 31; ++i) {
        s << ((block & mask) ? '1' : '0');
        mask <<= 1;
      }
    } else {
      auto bit = (block & (0x1 << 30)) ? '1' : '0';
      auto times = (block & (~(0x3 << 30)));
      for (uint32_t i = 0; i < times; ++i) {
        for (uint32_t j = 0; j < 31; ++j) {
          s << bit;
        }
      }
    }
  }
  auto res = s.str();
  res.erase(res.find_last_not_of("0") + 1);
  return res;
}

// Reduce by key for OR operation

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
  assert(keys.size() == vals.size());
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
vector<T> exclusive_scan(const vector<T>& vals) {
  vector<T> results(vals.size());
  results[0] = 0;
  for (size_t i = 1; i < vals.size(); ++i) {
    results[i] = results[i - 1] + vals[i - 1];
  }
  return results;
}

template<class T>
size_t stream_compaction(vector<T>& index, T val = 0) {
  index.erase(remove(begin(index), end(index), val), end(index));
  return index.size();
}

// WAH Algorithm ...

// in : input
// out: input, rids (both sorted by input)
void sort_rids_by_value(vector<uint32_t>& input, vector<uint32_t>& rids) {
  assert(input.size() == rids.size());
  iota(begin(rids), end(rids), 0);
  for (size_t i = (input.size() - 1); i > 0; --i) {
    for (size_t j = 0; j < i; ++j) {
      if (input[j] > input[j + 1]) {
        // switch input
        auto tmp     = input[j];
        input[j    ] = input[j + 1];
        input[j + 1] = tmp;
        // switch rids
        tmp         = rids[j    ];
        rids[j    ] = rids[j + 1];
        rids[j + 1] = tmp;
      }
    }
  }
}

// in : rids, n (length)
// out: chids, lits
void produce_chunk_id_literals(vector<uint32_t>& rids,
                                vector<uint32_t>& chids,
                                vector<uint32_t>& lits) {
  assert(rids.size() == chids.size());
  assert(rids.size() == lits.size());
  for (size_t i = 0; i < rids.size(); ++i) {
    lits[i]  = 0x1 << (rids[i] % 31);
    lits[i] |= 0x1 << 31;
    chids[i] = rids[i] / 31;
  }
}

// in : input, chids, lits, n (length)
// out: input, chids, lits but reduced to length k
size_t merged_lit_by_val_chids(vector<uint32_t>& input,
                               vector<uint32_t>& chids,
                               vector<uint32_t>& lits) {
  return reduce_by_key(input, chids, lits);
}

// in : input, chids, k (reduced length)
// out: chids with 0-fill symbols
void produce_fills(vector<uint32_t>& input,
                   vector<uint32_t>& chids,
                   size_t k) {
  vector<uint32_t> heads(k);
  adjacent_difference(begin(input), begin(input) + k, begin(heads));
  heads.front() = 1;
  vector<uint32_t> new_chids(k);
  for (size_t i = 0; i < k; ++i) {
    if (heads[i] == 0) {
      new_chids[i] = chids[i] - chids[i - 1] - 1;
    } else {
      new_chids[i] = chids[i];
      // This seems to result in the loss of fills
      //if (chids[i] != 0)
      //  new_chids[i] = chids[i] - 1;
    }
  }
  chids = std::move(new_chids);
}

// in : chids, lits, k
// out: index, index_length
size_t fuse_fill_literals(vector<uint32_t>& chids,
                          vector<uint32_t>& lits,
                          vector<uint32_t>& index,
                          size_t k) {
  assert(chids.size() == k);
  assert(lits.size() == k);
  assert(index.size() >= 2*k);
  for (size_t i = 0; i < k; ++i) {
    index[2 * i] = chids[i];
    index[2 * i + 1] = lits[i];
  }
  return stream_compaction(index, 0u);
}

// in : chids, input, n
// out: keycnt, offsets
size_t compute_colum_length(vector<uint32_t>& input,
                            vector<uint32_t>& chids,
                            vector<uint32_t>& offsets,
                            size_t k) {
  vector<uint32_t> tmp(k);
  for (size_t i = 0; i < k; ++i) {
    tmp[i] = (1 + (chids[i] == 0 ? 0 : 1));
  }
  auto keycnt = reduce_by_key(input, tmp);
  offsets = exclusive_scan(tmp);
  return keycnt;
}

class config : public actor_system_config {
public:
  string filename = "";
  uint32_t bound = 0;
  bool print_results;
  config() {
    opt_group{custom_options_, "global"}
    .add(filename, "data-file,f", "File with test data (one value per line)")
    .add(bound, "bound,b", "maximum value (0 will scan values)")
    .add(print_results, "print,p", "print resulting bitmap index");
  }
};

} // namespace <anonymous>

void caf_main(actor_system&, const config& cfg) {
  vector<uint32_t> values;
  if (cfg.filename.empty()) {
    values = {10,  7, 22,  6,  7,  1,  9, 42,  2,  5,
              13,  3,  2,  1,  0,  1, 18, 18,  3, 13,
               5,  9,  0,  3,  2, 19,  5, 23, 22, 10,
               6, 22};
  } else {
    cout << "Reading data from '" << cfg.filename << "' ... " << flush;
    ifstream source{cfg.filename, std::ios::in};
    uint32_t next;
    while (source >> next) {
      values.push_back(next);
    }
  }
  auto amount = values.size();
  cout << "'" << amount << "' values." << endl;
  auto bound = cfg.bound;
  if (bound == 0 && amount > 0) {
    auto itr = max_element(values.begin(), values.end());
    bound = *itr;
  }
  cout << "Maximum value is '" << bound << "'." << endl;

  auto input = values;
  vector<uint32_t> rids(input.size());
  vector<uint32_t> chids(input.size());
  vector<uint32_t> lits(input.size());

  sort_rids_by_value(input, rids);
  produce_chunk_id_literals(rids, chids, lits);
  auto k = merged_lit_by_val_chids(input, chids, lits);
  cout << "Merged to " << k << " values" << endl;
  produce_fills(input, chids, k);
  vector<uint32_t> index(2 * k);
  auto index_length = fuse_fill_literals(chids, lits, index, k);
  vector<uint32_t> offsets(k);
  auto keycnt = compute_colum_length(input, chids, offsets, k);
  cout << "Index has " << index_length << " elements with "
      << keycnt << " keys" << endl;
  if (cfg.print_results) {
    uint32_t next = 0;
    for (size_t i = 0; i < keycnt; ++i) {
      auto key = input[i];
      auto offset = offsets[i];
      auto length = (i == keycnt - 1) ? index_length - offsets[i]
                                      : offsets[i + 1] - offsets[i];
      vector<uint32_t> tmp{index.begin() + offset,
                           index.begin() + offset + length};
      /*
      cout << "Accessing " << key << " from " << offset
           << " to " << (offset + length) << " (" << length << " blocks, key "
           << (i + 1) << " of " << keycnt << ")" << endl;
      */
      if (key != next) {
        for (uint32_t j = next; j < key; ++j) {
          cout << j << "\t" << endl;
        }
      }
      cout << key << '\t' << decoded_bitmap(tmp) << endl;
      next = key + 1;
    }
  }
}

CAF_MAIN()
