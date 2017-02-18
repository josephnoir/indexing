
#include <cmath>
#include <chrono>
#include <random>
#include <vector>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <unordered_map>

#include "caf/all.hpp"
#include "caf/opencl/all.hpp"
#include "caf/opencl/mem_ref.hpp"
#include "caf/opencl/actor_facade_phase.hpp"

using namespace std;
using namespace std::chrono;
using namespace caf;
using namespace caf::opencl;

// required to allow sending mem_ref<int> in messages
namespace caf {
  template <>
  struct allowed_unsafe_message_type<mem_ref<uint32_t>> : std::true_type {};
}

namespace {

using vec = std::vector<uint32_t>;
using val = vec::value_type;

using add_atom = atom_constant<atom("add")>;
using init_atom = atom_constant<atom("init")>;
using quit_atom = atom_constant<atom("quit")>;
using index_atom = atom_constant<atom("index")>;

constexpr const char* kernel_file_01 = "./include/sort_rids_by_value.cl";
constexpr const char* kernel_file_02 = "./include/produce_chunk_id_literals.cl";
constexpr const char* kernel_file_03 = "./include/merge_lit_by_val_chids.cl";
constexpr const char* kernel_file_04 = "./include/produce_fills.cl";
constexpr const char* kernel_file_05 = "./include/fuse_fill_literals.cl";
/*
constexpr const char* kernel_file_06 = "./include/compute_colum_length.cl";
*/
constexpr const char* kernel_file_07 = "./include/stream_compaction.cl";

//constexpr const char* kernel_name_01a = "kernel_wah_index";

/*****************************************************************************\
                              JUST FOR OUTPUT
\*****************************************************************************/

template <class T>
void valid_or_exit(T expr, const std::string& str = "") {
  if (expr)
    return;
  if (str.empty()) {
    cout << "[!!!] Something went wrong" << endl;
  } else {
    cout << "[!!!] " << str << endl;
  }
  exit(-1);
}

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

/*
string decoded_bitmap(const vec& bitmap) {
  if (bitmap.empty()) {
    return "";
  }
  stringstream s;
  for (auto& block : bitmap) {
    if (block & (0x1 << 31)) {
      val mask = 0x1;
      for (int i = 0; i < 31; ++i) {
        s << ((block & mask) ? '1' : '0');
        mask <<= 1;
      }
    } else {
      auto bit = (block & (0x1 << 30)) ? '1' : '0';
      auto times = (block & (~(0x3 << 30)));
      for (val i = 0; i < times; ++i) {
        for (val j = 0; j < 31; ++j) {
          s << bit;
        }
      }
    }
  }
  auto res = s.str();
  //auto tmp = res.size();
  //s.str(string());
  //s << tmp;
  //return s.str();
  res.erase(res.find_last_not_of("0") + 1);
  return res;
}
*/

/*****************************************************************************\
        TESTS FUNCTIONS ON CPU FOR COMPARISON (TODO: DELTE THIS LATER)
\*****************************************************************************/

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

template<class T>
size_t stream_compaction(vector<T>& index, T val = 0) {
  index.erase(remove(begin(index), end(index), val), end(index));
  return index.size();
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

/*****************************************************************************\
                          INTRODUCE SOME CLI ARGUMENTS
\*****************************************************************************/

class config : public actor_system_config {
public:
  string filename = "";
  val bound = 0;
  string device_name = "GeForce GTX 780M";
  bool print_results;
  config() {
    load<opencl::manager>();
    opt_group{custom_options_, "global"}
    .add(filename, "data-file,f", "file with test data (one value per line)")
    .add(bound, "bound,b", "maximum value (0 will scan values)")
    .add(device_name, "device,d", "device for computation (GeForce GTX 780M, "
                      "empty string will take first available device)")
    .add(print_results, "print,p", "print resulting bitmap index");
  }
};

} // namespace <anonymous>

/*****************************************************************************\
                                    MAIN!
\*****************************************************************************/

void caf_main(actor_system& system, const config& cfg) {
  vec values;
  if (cfg.filename.empty()) {
    values = {10,  7, 22,  6,  7,  1,  9, 42,  2,  5,
              13,  3,  2,  1,  0,  1, 18, 18,  3, 13,
               5,  9,  0,  3,  2, 19,  5, 23, 22, 10,
               6, 22};
  } else {
    cout << "Reading data from '" << cfg.filename << "' ... " << flush;
    ifstream source{cfg.filename, std::ios::in};
    val next;
    while (source >> next) {
      values.push_back(next);
    }
  }
  cout << "'" << values.size() << "' values." << endl;
  auto bound = cfg.bound;
  if (bound == 0 && !values.empty()) {
    auto itr = max_element(values.begin(), values.end());
    bound = *itr;
  }
  cout << "Maximum value is '" << bound << "'." << endl;

  auto& mngr = system.opencl_manager();

  // get device
  auto opt = mngr.get_device_if([&](const device& dev) {
      if (cfg.device_name.empty())
        return true;
      return dev.get_name() == cfg.device_name;
  });
  if (!opt) {
    cerr << "Device " << cfg.device_name << " not found." << endl;
    return;
  } else {
    cout << "Using device named '" << opt->get_name() << "'." << endl;
  }
  auto dev = *opt;

  // Create test data
  auto input = values;
  vec rids(input.size());
  vec chids(input.size());
  vec lits(input.size());
  sort_rids_by_value(input, rids);
  produce_chunk_id_literals(rids, chids, lits);
  auto k_test = merged_lit_by_val_chids(input, chids, lits);
  vec chids_produce{chids};
  produce_fills(input, chids_produce, k_test);
  vec index(2 * k_test);
  vec chids_fuse{chids_produce};
  auto index_length = fuse_fill_literals(chids_fuse, lits, index, k_test);
  vec offsets(k_test);
  vec input_col{input};
  auto keycnt = compute_colum_length(input_col, chids_fuse, offsets, k_test);
  cout << "Created test data." << endl;
  cout << "Index has " << index_length << " elements with "
       << keycnt << " keys." << endl;

  // load kernels
  auto prog_rids   = mngr.create_program_from_file(kernel_file_01, "", dev);
  auto prog_chunks = mngr.create_program_from_file(kernel_file_02, "", dev);
  auto prog_merge  = mngr.create_program_from_file(kernel_file_03, "", dev);
  auto prog_fills  = mngr.create_program_from_file(kernel_file_04, "", dev);
  auto prog_fuse   = mngr.create_program_from_file(kernel_file_05, "", dev);
  /*
  auto prog_colum  = mngr.create_program_from_file(kernel_file_06, "", dev);
  */
  auto prog_sc     = mngr.create_program_from_file(kernel_file_07, "", dev);

  // create spawn configuration
  auto n = values.size();
  auto wgs = dev.get_max_compute_units();
  auto index_space      = spawn_config{dim_vec{n}};
  //auto index_space_half = spawn_config{dim_vec{n / 2}};
  auto index_space_128  = spawn_config{dim_vec{n}, {}, dim_vec{128}};

  // buffers for execution
  vec config{static_cast<val>(n)};
  auto inpt_ref = dev.global_buffer(buffer_type::input_output, values);
  // TODO: should be scratch space, but output is useful for testing
  auto chid_ref = dev.scratch_space<val>(n, buffer_type::output);
  auto lits_ref = dev.scratch_space<val>(n, buffer_type::output);
  auto temp_ref = dev.scratch_space<val>(n, buffer_type::output);
  {
    // create phases
    auto rids_1 = mngr.spawn_phase<vec, vec, vec>(prog_rids, "create_rids",
                                                  index_space);
    /*
    auto rids_2 = mngr.spawn_phase<vec, vec, vec>(prog_rids,
                                                  "ParallelBitonic_B2",
                                                  index_space_half);
    */
    auto rids_3 = mngr.spawn_phase<vec, vec, vec, vec>(prog_rids,
                                                       "ParallelSelection",
                                                       index_space);
    auto chunks = mngr.spawn_phase<vec, vec, vec>(prog_chunks,
                                                  "produce_chunks",
                                                  index_space);
    auto merge_heads = mngr.spawn_phase<vec, vec, vec>(prog_merge,
                                                       "create_heads",
                                                       index_space);
    auto merge_scan = mngr.spawn_phase<vec,vec>(prog_merge,
                                                "lazy_segmented_scan",
                                                index_space);
    auto sc_count = mngr.spawn_phase<vec,vec,vec,vec>(prog_sc,
                                                      "countElts",
                                                      index_space_128);
    auto sc_move = mngr.spawn_phase<vec,vec,vec,vec,
                                    vec,vec,vec,vec>(prog_sc,
                                                     "moveValidElementsStaged",
                                                     index_space_128);
    auto fills = mngr.spawn_phase<vec,vec,vec,vec>(prog_fills,
                                                   "produce_fills",
                                                   index_space);
    auto fuse_prep = mngr.spawn_phase<vec,vec,vec,vec>(prog_fuse,
                                                       "prepare_index",
                                                       index_space);
    // kernel executions
    // temp_ref used as rids buffer
    scoped_actor self{system};
    auto conf_ref = dev.global_buffer(buffer_type::input, config);
    self->send(rids_1, conf_ref, inpt_ref, temp_ref);
    self->receive(
      [&](mem_ref<val>&, mem_ref<val>&, mem_ref<val>&) {
        // nop
      }
    );
    config.resize(2);
    // chids and lit only used as temporary buffers
    self->send(rids_3, inpt_ref, temp_ref, chid_ref, lits_ref);
    self->receive(
      [&](mem_ref<val>&, mem_ref<val>&, mem_ref<val>&, mem_ref<val>&) {
        // nop
      }
    );
    inpt_ref = chid_ref;
    temp_ref = lits_ref;
    chid_ref = dev.scratch_space<val>(n, buffer_type::output);
    lits_ref = dev.scratch_space<val>(n, buffer_type::output);
    /*
    for (val length = 1; length < values.size(); length <<= 1) {
      int inc = length;
      bool done = false;
      config[0] = inc;
      config[1] = length << 1;
      conf_ref = dev.global_buffer(buffer_type::input, config);
      self->send(rids_2, conf_ref, inpt_ref, temp_ref);
      self->receive_while([&] { return !done; })(
        [&](mem_ref<val>& conf, mem_ref<val>& vals,
            mem_ref<val>& rids) {
          inc >>= 1;
          if (inc > 0) {
            config[0] = inc;
            conf = dev.global_buffer(buffer_type::input, config);
            self->send(rids_2, conf, vals, rids);
          } else {
            done = true;
          }
        }
      );
    }
    */
    cout << "DONE: sort_rids_by_value" << endl;
    /*
    auto inpt_exp = inpt_ref.data();
    auto rids_exp = temp_ref.data();
    if (!inpt_exp || !rids_exp)
      cout << "Something went wrong" << endl;
    else {
      auto inp = *inpt_exp;
      auto rid = *rids_exp;
      for (size_t i = 0; i < inp.size(); ++i) {
        cout << "[" << (inp[i] == input[i]) << "|" << (rid[i] == rids[i]) << "] "
             << setw(4) << inp[i] << ": "
             << setw(4) << rid[i] << " =?= " << setw(4) << rids[i] << endl;
      }
    }
    */
    self->send(chunks, temp_ref, chid_ref, lits_ref);
    self->receive(
      [&](mem_ref<val>&, mem_ref<val>& chid_r, mem_ref<val>& lit_r) {
        cout << "DONE: produce_chunk_id_literals" << endl;
        auto in_exp = inpt_ref.data();
        auto ch_exp = chid_r.data();
        auto li_exp = lit_r.data();
        valid_or_exit(in_exp);
        valid_or_exit(ch_exp);
        valid_or_exit(li_exp);
        auto in = *in_exp;
        auto ch = *ch_exp;
        auto li = *li_exp;
        cout << "Input equal: " << (in == input) << endl;
        cout << "Chids equal: " << (ch == chids) << endl;
        cout << "Lits  equal: " << (li == lits) << endl;
      }
    );
    // use temp as heads array
    self->send(merge_heads, inpt_ref, chid_ref, temp_ref);
    self->receive(
      [&](mem_ref<val>&, mem_ref<val>&, mem_ref<val>&) {
        cout << "Created heads array" << endl;
      }
    );
    self->send(merge_scan, temp_ref, lits_ref);
    self->receive(
      [&](mem_ref<val>&, mem_ref<val>&) {
        cout << "Merged values" << endl;
        /*
        auto res1 = heads.data();
        auto res2 = lits.data();
        if (!res1 || !res2) {
          cout << "Something went wrong!" << endl;
        } else {
          for (size_t i = 0; i < res1->size(); ++i) {
            cout << as_binary(res1->at(i)) << " : "
                 << as_binary(res2->at(i)) << endl;
          }
        }
        */
      }
    );
    // stream compact inpt, chid, lits by heads value
    config[0] = static_cast<val>(n);
    config[1] = 0;
    conf_ref = dev.global_buffer(buffer_type::input_output, config);
    auto blocks_ref = dev.scratch_space<val>(wgs, buffer_type::output);
    auto b128_ref = dev.local_buffer<val>(buffer_type::scratch_space, 128);
    self->send(sc_count, conf_ref, blocks_ref, temp_ref, b128_ref);
    self->receive(
      [&](mem_ref<val>&, mem_ref<val>&, mem_ref<val>&, mem_ref<val>&) {
        cout << "Count step done." << endl;
      }
    );
    auto out_ref = dev.scratch_space<val>(n, buffer_type::output);
    self->send(sc_move, conf_ref,   inpt_ref, out_ref,  temp_ref,
                        blocks_ref, b128_ref, b128_ref, b128_ref);
    self->receive(
      [&](mem_ref<val>&, mem_ref<val>&, mem_ref<val>&, mem_ref<val>&,
          mem_ref<val>&, mem_ref<val>&, mem_ref<val>&, mem_ref<val>&) {
        cout << "Merge step done (input)." << endl;
      }
    );
    inpt_ref.swap(out_ref);
    self->send(sc_move, conf_ref, chid_ref, out_ref, temp_ref,
                            blocks_ref, b128_ref, b128_ref, b128_ref);
    self->receive(
      [&](mem_ref<val>&, mem_ref<val>&, mem_ref<val>&, mem_ref<val>&,
          mem_ref<val>&, mem_ref<val>&, mem_ref<val>&, mem_ref<val>&) {
        cout << "Merge step done (chids)." << endl;
      }
    );
    chid_ref.swap(out_ref);
    self->send(sc_move, conf_ref, lits_ref, out_ref, temp_ref,
                            blocks_ref, b128_ref, b128_ref, b128_ref);
    self->receive(
      [&](mem_ref<val>&, mem_ref<val>&, mem_ref<val>&, mem_ref<val>&,
          mem_ref<val>&, mem_ref<val>&, mem_ref<val>&, mem_ref<val>&) {
        cout << "Merge step done (lits)." << endl;
      }
    );
    lits_ref.swap(out_ref);
    cout << "DONE: merged_lit_by_val_chids." << endl;
    auto res_conf = conf_ref.data();
    auto res_inpt = inpt_ref.data();
    auto res_chid = chid_ref.data();
    auto res_lits = lits_ref.data();
    valid_or_exit(res_conf);
    valid_or_exit(res_inpt);
    valid_or_exit(res_chid);
    valid_or_exit(res_lits);
    auto k = res_conf->at(1);
    valid_or_exit(k == k_test);
    vec new_inpt{*res_inpt};
    vec new_chid{*res_chid};
    vec new_lits{*res_lits};
    new_inpt.resize(k);
    new_chid.resize(k);
    new_lits.resize(k);
    valid_or_exit(new_inpt == input, "input not equal");
    valid_or_exit(new_chid == chids, "chids not equal");
    valid_or_exit(new_lits == lits, "lits not equal");
    // we should reconfigure the NDRange of fills-actor here to k
    self->send(fills, conf_ref, inpt_ref, chid_ref, out_ref);
    self->receive(
      [&](mem_ref<val>&, mem_ref<val>&, mem_ref<val>&, mem_ref<val>& ) {
        cout << "DONE: produce fills." << endl;
      }
    );
    chid_ref.swap(out_ref);
    res_inpt = inpt_ref.data();
    res_chid = chid_ref.data();
    valid_or_exit(res_inpt, "destroyed input");
    valid_or_exit(res_chid, "can't read fills");
    new_inpt = *res_inpt;
    new_chid = *res_chid;
    new_inpt.resize(k);
    new_chid.resize(k);
    valid_or_exit(new_inpt == input, "input not equal");
    valid_or_exit(new_chid == chids_produce, "chids not equal");
    // TODO: release buffers no longer needed
    // we should reconfigure the NDRange of fuse_prep-actor here to k
    auto idx_ref = dev.scratch_space<val>(2*k, buffer_type::output);
    self->send(fuse_prep, conf_ref, chid_ref, lits_ref, idx_ref);
    self->receive(
      [&](mem_ref<val>&, mem_ref<val>&, mem_ref<val>&, mem_ref<val>& ) {
        cout << "Prepared index." << endl;
      }
    );
    // stream compaction using input ad valid
    // currently newly created actor to change NDRange
    //auto wi = 128 * (((2 * k) / 128) + (((2 * k) % 128) ? 1 : 0));
    auto wi = ((2 * k) + 128 - 1) & ~(128 - 1); // only for powers of 2
    cout << "wi = " << wi << endl;
    auto index_space_k_128 = spawn_config{dim_vec{wi}, {}, dim_vec{128}};
    sc_count = mngr.spawn_phase<vec,vec,vec,vec>(prog_sc, "countElts",
                                                 index_space_k_128);
    sc_move = mngr.spawn_phase<vec,vec,vec,vec,
                               vec,vec,vec,vec>(prog_sc,
                                                "moveValidElementsStaged",
                                                index_space_k_128);
    cout << "count for idx" << endl;
    config[0] = k;
    config[1] = 0;
    conf_ref = dev.global_buffer(buffer_type::input_output, config);
    self->send(sc_count, conf_ref, blocks_ref, idx_ref, b128_ref);
    self->receive(
      [&](mem_ref<val>&, mem_ref<val>&, mem_ref<val>&, mem_ref<val>&) {
        cout << "Count step done." << endl;
      }
    );
    out_ref = dev.scratch_space<val>(2 * k, buffer_type::output);
    self->send(sc_move, conf_ref,   idx_ref,  out_ref,  idx_ref,
                        blocks_ref, b128_ref, b128_ref, b128_ref);
    self->receive(
      [&](mem_ref<val>&, mem_ref<val>&, mem_ref<val>&, mem_ref<val>&,
          mem_ref<val>&, mem_ref<val>&, mem_ref<val>&, mem_ref<val>&) {
        cout << "Merge step done (input)." << endl;
      }
    );
    idx_ref.swap(out_ref);
    cout << "DONE: fuse_fill_literals." << endl;
    auto idx_conf = conf_ref.data();
    /*
    valid_or_exit(res_conf, "Can't read conf after stream compcation.");
    auto idx_len = res_conf->at(1);
    cout << "Created index of length " << idx_len << endl;
    */
  }
  // clean up
  system.await_all_actors_done();
}

CAF_MAIN()
