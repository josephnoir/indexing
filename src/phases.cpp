
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

#define SHOW_TIME_CONSUMPTION

using namespace std;
using namespace std::chrono;
using namespace caf;
using namespace caf::opencl;

struct radix_config {
  uint32_t radices;  // number of radices
  uint32_t blocks;   // number of blocks
  uint32_t gpb;      // groups per block
  uint32_t tpg;      // threads per group
  uint32_t epg;      // elements per group
  uint32_t rpb;      // radices per block
  uint32_t mask;     // bit mask
  uint32_t l_val;    // L
  uint32_t tpb;      // threads per block
  uint32_t size;     // total elements
};

namespace {

using uval = uint32_t;
using uvec = std::vector<uval>;
using uref = mem_ref<uval>;

using add_atom = atom_constant<atom("add")>;
using init_atom = atom_constant<atom("init")>;
using quit_atom = atom_constant<atom("quit")>;
using index_atom = atom_constant<atom("index")>;

constexpr const char* kernel_file_01 = "./include/sort_rids_by_value.cl";
constexpr const char* kernel_file_02 = "./include/produce_chunk_id_literals.cl";
constexpr const char* kernel_file_03 = "./include/merge_lit_by_val_chids.cl";
constexpr const char* kernel_file_04 = "./include/produce_fills.cl";
constexpr const char* kernel_file_05 = "./include/fuse_fill_literals.cl";
constexpr const char* kernel_file_06 = "./include/compute_colum_length.cl";
constexpr const char* kernel_file_07 = "./include/stream_compaction.cl";
constexpr const char* kernel_file_08 = "./include/scan.cl";
constexpr const char* kernel_file_09 = "./include/radix.cl";

//constexpr const char* kernel_name_01a = "kernel_wah_index";

} // namespace anonymous

// required to allow sending mem_ref<int> in messages
namespace caf {
  template <>
  struct allowed_unsafe_message_type<uref> : std::true_type {};
  template <>
  struct allowed_unsafe_message_type<mem_ref<radix_config>>
    : std::true_type {};
}


namespace {

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
string decoded_bitmap(const uvec& bitmap) {
  if (bitmap.empty()) {
    return "";
  }
  stringstream s;
  for (auto& block : bitmap) {
    if (block & (0x1 << 31)) {
      uval mask = 0x1;
      for (int i = 0; i < 31; ++i) {
        s << ((block & mask) ? '1' : '0');
        mask <<= 1;
      }
    } else {
      auto bit = (block & (0x1 << 30)) ? '1' : '0';
      auto times = (block & (~(0x3 << 30)));
      for (uval i = 0; i < times; ++i) {
        for (uval j = 0; j < 31; ++j) {
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
/*
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
*/

template <class T, class E = typename enable_if<is_integral<T>::value>::type>
T round_up(T numToRound, T multiple)  {
  assert(multiple > 0);
  return ((numToRound + multiple - 1) / multiple) * multiple;
}

/*****************************************************************************\
                          INTRODUCE SOME CLI ARGUMENTS
\*****************************************************************************/

class config : public actor_system_config {
public:
  string filename = "";
  uval bound = 0;
  int loops = 1;
  string device_name = "GeForce GT 650M";
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
  uvec values;
  if (cfg.filename.empty()) {
    cout << "A filename with test data is required, see --help for detailts."
         << endl;
    return;
  }
  //cout << "Reading data from '" << cfg.filename << "' ... " << flush;
  ifstream source{cfg.filename, std::ios::in};
  uval next;
  while (source >> next) {
    values.push_back(next);
  }
  //cout << "values: " << values.size() << endl;
  auto bound = cfg.bound;
  if (bound == 0 && !values.empty()) {
    auto itr = max_element(values.begin(), values.end());
    bound = *itr;
  }
  //cout << "Maximum value is '" << bound << "'." << endl;

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
  } /*else {
    cout << "Using device named '" << opt->get_name() << "'." << endl;
  }*/
  auto dev = *opt;

  // Create test data
  /*
  auto input = values;
  uvec rids(input.size());
  uvec chids(input.size());
  uvec lits(input.size());
  sort_rids_by_value(input, rids);
  produce_chunk_id_literals(rids, chids, lits);
  auto k_test = merged_lit_by_val_chids(input, chids, lits);
  uvec chids_produce{chids};
  produce_fills(input, chids_produce, k_test);
  uvec index(2 * k_test);
  uvec chids_fuse{chids_produce};
  auto index_length = fuse_fill_literals(chids_fuse, lits, index, k_test);
  uvec offsets(k_test);
  uvec input_col{input};
  auto keycnt = compute_colum_length(input_col, chids_fuse, offsets, k_test);
  cout << "Created test data." << endl;
  cout << "Index has " << index_length << " elements with "
       << keycnt << " keys." << endl;
  */
  // load kernels
  auto prog_rids   = mngr.create_program_from_file(kernel_file_01, "", dev);
  auto prog_chunks = mngr.create_program_from_file(kernel_file_02, "", dev);
  auto prog_merge  = mngr.create_program_from_file(kernel_file_03, "", dev);
  auto prog_fills  = mngr.create_program_from_file(kernel_file_04, "", dev);
  auto prog_fuse   = mngr.create_program_from_file(kernel_file_05, "", dev);
  auto prog_colum  = mngr.create_program_from_file(kernel_file_06, "", dev);
  auto prog_sc     = mngr.create_program_from_file(kernel_file_07, "", dev);
  auto prog_es     = mngr.create_program_from_file(kernel_file_08, "", dev);
  auto prog_radix  = mngr.create_program_from_file(kernel_file_09, "", dev);

  // create spawn configuration
  auto n = values.size();
  auto index_space = spawn_config{dim_vec{n}};
  auto wi = round_up(n, 128ul);
  auto index_space_128  = spawn_config{dim_vec{wi}, {}, dim_vec{128}};

  // buffers we alread know we need
  uvec config{static_cast<uval>(n)};
  auto inpt_ref = dev.global_argument(values);
  auto chid_ref = dev.scratch_argument<uval>(n, buffer_type::output);
  auto lits_ref = dev.scratch_argument<uval>(n, buffer_type::output);
  auto temp_ref = dev.scratch_argument<uval>(n, buffer_type::output);

  // sort configuration
  // thread block has multiple thread groups has multiple threads
  // - blocks match to OpenCL work groups
  // - threads map to work items, but ids are counted inside a block
  // - groups separate threads of a block into multiple bundles
  // TODO: Optimized runs with regard to cardinality
  uint32_t cardinality = 16;
  uint32_t l_val = 4; // bits used as a bucket in each radix iteration
  uint32_t radices = 1 << l_val;
  uint32_t blocks
    = (dev.get_max_compute_units() <= (radices / 2)) ? (radices / 2) : radices;
  uint32_t threads_per_block
    = max(radices, static_cast<uint32_t>(dev.get_max_work_group_size()));
  uint32_t threads_per_group = threads_per_block / radices;
  uint32_t groups_per_block = threads_per_block / threads_per_group;
  uint32_t mask = (1 << l_val) - 1;
  uint32_t radices_per_block = radices / blocks;
  uint32_t elements = static_cast<uint32_t>(n);
  uint32_t groups = groups_per_block * blocks;
  uint32_t elements_per_group = (elements / groups) + 1;
  // groups share a counter and each group requires a counter per radix
  uint32_t counters = radices * groups_per_block * blocks;
  uint32_t prefixes = radices;
  radix_config rc = {
    radices,
    blocks,
    groups_per_block,
    threads_per_group,
    elements_per_group,
    radices_per_block,
    mask,
    l_val,
    threads_per_block,
    elements
  };
  size_t radix_global = threads_per_block * blocks;
  size_t radix_local = threads_per_block;
  auto radix_range = spawn_config{dim_vec{radix_global}, {},
                                  dim_vec{radix_local}};
  auto zero_range = spawn_config{dim_vec{counters}};
  {
    auto start = high_resolution_clock::now();
    // create phases
    auto rids_1 = mngr.spawn_stage<uvec, uvec, uvec>(prog_rids, "create_rids",
                                                  index_space);
    auto chunks = mngr.spawn_stage<uvec, uvec, uvec>(prog_chunks,
                                                  "produce_chunks",
                                                  index_space);
    auto merge_heads = mngr.spawn_stage<uvec, uvec, uvec>(prog_merge,
                                                       "create_heads",
                                                       index_space);
    auto merge_scan = mngr.spawn_stage<uvec,uvec>(prog_merge,
                                                "lazy_segmented_scan",
                                                index_space);
    auto sc_count = mngr.spawn_stage<uvec,uvec,uvec,uvec>(prog_sc,
                                                      "countElts",
                                                      index_space_128);
    auto sc_move = mngr.spawn_stage<uvec,uvec,uvec,uvec,
                                    uvec,uvec,uvec,uvec>(prog_sc,
                                                     "moveValidElementsStaged",
                                                     index_space_128);
    auto fills = mngr.spawn_stage<uvec,uvec,uvec,uvec>(prog_fills,
                                                   "produce_fills",
                                                   index_space);
    auto fuse_prep = mngr.spawn_stage<uvec,uvec,uvec,uvec>(prog_fuse,
                                                       "prepare_index",
                                                       index_space);
    auto radix_zero = mngr.spawn_stage<uvec>(prog_radix, "zeroes", zero_range);
    auto radix_count = mngr.spawn_stage<uvec,uvec,radix_config,
                                        uval>(prog_radix, "count", radix_range);
    auto radix_sum = mngr.spawn_stage<uvec,uvec,uvec,uvec,radix_config,
                                      uval>(prog_radix, "scan", radix_range);
    auto radix_move
      = mngr.spawn_stage<uvec,uvec,uvec,uvec,uvec,uvec,uvec,uvec,
                         radix_config, uval>(prog_radix, "reorder_kv",
                                            radix_range);
#ifdef SHOW_TIME_CONSUMPTION
    auto to = high_resolution_clock::now();
    auto from = high_resolution_clock::now();
#endif

    // kernel executions
    // temp_ref used as rids buffer
    scoped_actor self{system};
    auto conf_ref = dev.global_argument(config);
    self->send(rids_1, conf_ref, inpt_ref, temp_ref);
    self->receive(
      [&](uref&, uref&, uref&) {
        // nop
      }
    );
    /*
    // chids and lit only used as temporary buffers
    self->send(rids_3, inpt_ref, temp_ref, chid_ref, lits_ref);
    self->receive(
      [&](uref&, uref&, uref&, uref&) {
        // nop
      }
    );
    inpt_ref = chid_ref;
    temp_ref = lits_ref;
    chid_ref = dev.scratch_argument<uval>(n, buffer_type::output);
    lits_ref = dev.scratch_argument<uval>(n, buffer_type::output);
    */
    /*
    for (val length = 1; length < values.size(); length <<= 1) {
      int inc = length;
      bool done = false;
      config.resize(2);
      config[0] = inc;
      config[1] = length << 1;
      conf_ref = dev.global_argument(config, buffer_type::input);
      self->send(rids_2, conf_ref, inpt_ref, temp_ref);
      self->receive_while([&] { return !done; })(
        [&](uref& conf, uref& vals, uref& rids) {
          inc >>= 1;
          if (inc > 0) {
            config[0] = inc;
            conf = dev.global_argument(config, buffer_type::input);
            self->send(rids_2, conf, vals, rids);
          } else {
            done = true;
          }
        }
      );
    }
    */
    {
      // radix sort for values by key using inpt as keys and temp as values
      auto r_keys_in = inpt_ref;
      auto r_keys_out = chid_ref;
      auto r_values_in = temp_ref;
      auto r_values_out = lits_ref;
      auto r_counters = dev.scratch_argument<uval>(counters);
      auto r_prefixes = dev.scratch_argument<uval>(prefixes);
      auto r_local_a = dev.local_argument<uval>(groups_per_block * blocks);
      auto r_local_b = dev.local_argument<uval>(groups_per_block * radices);
      auto r_local_c = dev.local_argument<uval>(radices);
      auto r_conf = dev.private_argument(rc);
      uint32_t iterations = cardinality / l_val;
      for (uint32_t i = 0; i < iterations; ++i) {
        auto r_offset = dev.private_argument(static_cast<uval>(l_val * i));
        self->send(radix_zero, r_counters);
        self->receive([&](uref&) { });
        if (i > 0) {
          std::swap(r_keys_in, r_keys_out);
          std::swap(r_values_in, r_values_out);
        }
        self->send(radix_count, r_keys_in, r_counters, r_conf, r_offset);
        self->receive([&](uref&, uref&, mem_ref<radix_config>&,
                          uref&) { });
        self->send(radix_sum, r_keys_in, r_counters, r_prefixes, r_local_a,
                              r_conf, r_offset);
        self->receive([&](uref&, uref&, uref&,
                          uref&,
                          mem_ref<radix_config>&, uref&) { });
        self->send(radix_move, r_keys_in, r_keys_out,
                               r_values_in, r_values_out,
                               r_counters, r_prefixes,
                               r_local_b, r_local_c,
                               r_conf, r_offset);
        self->receive([&](uref&, uref&, uref&,
                          uref&, uref&, uref&,
                          uref&, uref&,
                          mem_ref<radix_config>&, uref&) { });
      }
      inpt_ref = r_keys_out;
      temp_ref = r_values_out;
      chid_ref = dev.scratch_argument<uval>(n, buffer_type::output);
      lits_ref = dev.scratch_argument<uval>(n, buffer_type::output);
    }
    //cout << "DONE: sort_rids_by_value" << endl;
#ifdef SHOW_TIME_CONSUMPTION
    to = high_resolution_clock::now();
    cout << duration_cast<microseconds>(to - from).count() << " us" << endl;
    from = high_resolution_clock::now();
#endif
    /*
    auto inpt_exp = inpt_ref.data();
    auto rids_exp = temp_ref.data();
    if (!inpt_exp) {
      cout << "Can't read keys back (" << system.render(inpt_exp.error())
           << ")." << endl;
    } else if (!rids_exp) {
      cout << "Can't read values back (" << system.render(rids_exp.error())
           << ")." << endl;
    } else {
      auto inp = *inpt_exp;
      auto rid = *rids_exp;
      vector<size_t> failed_keys;
      vector<size_t> failed_values;
      for (size_t i = 0; i < inp.size(); ++i) {
        if (inp[i] != input[i])
          failed_keys.push_back(i);
        if (rid[i] != rids[i])
          failed_values.push_back(i);
      }
      if (failed_keys.empty() && failed_values.empty())
        cout << "Success." << endl;
      else
        cout << "failed_keys for "
             << failed_keys.size() << " keys and "
             << failed_values.size() << " values." << endl;
    }
    return;
    */
    self->send(chunks, temp_ref, chid_ref, lits_ref);
    self->receive(
      [&](uref&, uref& /*chid_r*/, uref& /*lit_r*/) {
        /*
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
        */
      }
    );
#ifdef SHOW_TIME_CONSUMPTION
    to = high_resolution_clock::now();
    cout << duration_cast<microseconds>(to - from).count()<< " us" << endl;
    from = high_resolution_clock::now();
#endif
    // use temp as heads array
    self->send(merge_heads, inpt_ref, chid_ref, temp_ref);
    self->receive(
      [&](mem_ref<uval>&, mem_ref<uval>&, mem_ref<uval>&) {
       // cout << "Created heads array" << endl;
      }
    );
    self->send(merge_scan, temp_ref, lits_ref);
    self->receive(
      [&](mem_ref<uval>&, mem_ref<uval>&) {
        //cout << "Merged values" << endl;
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
    config.resize(2);
    config[0] = static_cast<uval>(n);
    config[1] = 0;
    conf_ref = dev.global_argument(config, buffer_type::input_output);
    auto blocks_size = wi / 128; // dev.get_max_compute_units();
    auto blocks_ref = dev.scratch_argument<uval>(blocks_size,
                                                 buffer_type::output);
    auto b128_ref = dev.local_argument<uval>(128);
    self->send(sc_count, conf_ref, blocks_ref, temp_ref, b128_ref);
    self->receive(
      [&](mem_ref<uval>&, mem_ref<uval>&, mem_ref<uval>&, mem_ref<uval>&) {
        //cout << "Count step done." << endl;
      }
    );
    auto out_ref = dev.scratch_argument<uval>(n, buffer_type::output);
    self->send(sc_move, conf_ref,   inpt_ref, out_ref,  temp_ref,
                        blocks_ref, b128_ref, b128_ref, b128_ref);
    self->receive(
      [&](mem_ref<uval>&, mem_ref<uval>&, mem_ref<uval>&, mem_ref<uval>&,
          mem_ref<uval>&, mem_ref<uval>&, mem_ref<uval>&, mem_ref<uval>&) {
        //cout << "Merge step done (input)." << endl;
      }
    );
    inpt_ref.swap(out_ref);
    self->send(sc_move, conf_ref, chid_ref, out_ref, temp_ref,
                        blocks_ref, b128_ref, b128_ref, b128_ref);
    self->receive(
      [&](mem_ref<uval>&, mem_ref<uval>&, mem_ref<uval>&, mem_ref<uval>&,
          mem_ref<uval>&, mem_ref<uval>&, mem_ref<uval>&, mem_ref<uval>&) {
        //cout << "Merge step done (chids)." << endl;
      }
    );
    chid_ref.swap(out_ref);
    self->send(sc_move, conf_ref, lits_ref, out_ref, temp_ref,
                        blocks_ref, b128_ref, b128_ref, b128_ref);
    self->receive(
      [&](uref&, uref&, uref&, uref&,
          uref&, uref&, uref&, uref&) {
        //cout << "Merge step done (lits)." << endl;
      }
    );
    lits_ref.swap(out_ref);
    // cout << "DONE: merge_lit_by_val_chids." << endl;
#ifdef SHOW_TIME_CONSUMPTION
    to = high_resolution_clock::now();
    cout << duration_cast<microseconds>(to - from).count()<< " us" << endl;
    from = high_resolution_clock::now();
#endif
    auto res_conf = conf_ref.data();
    /*
    auto res_inpt = inpt_ref.data();
    auto res_chid = chid_ref.data();
    auto res_lits = lits_ref.data();
    valid_or_exit(res_conf);
    valid_or_exit(res_inpt);
    valid_or_exit(res_chid);
    valid_or_exit(res_lits);
    */
    auto k = res_conf->at(1);
    /*
    valid_or_exit(k == k_test);
    uvec new_inpt{*res_inpt};
    uvec new_chid{*res_chid};
    uvec new_lits{*res_lits};
    new_inpt.resize(k);
    new_chid.resize(k);
    new_lits.resize(k);
    valid_or_exit(new_inpt == input, "input not equal");
    valid_or_exit(new_chid == chids, "chids not equal");
    valid_or_exit(new_lits == lits, "lits not equal");
    */
    // we should reconfigure the NDRange of fills-actor here to k
    self->send(fills, conf_ref, inpt_ref, chid_ref, out_ref);
    self->receive(
      [&](uref&, uref&, uref&, uref& ) {
        //cout << "DONE: produce fills." << endl;
      }
    );
#ifdef SHOW_TIME_CONSUMPTION
    to = high_resolution_clock::now();
    cout << duration_cast<microseconds>(to - from).count()<< " us" << endl;
    from = high_resolution_clock::now();
#endif
    chid_ref.swap(out_ref);
    /*
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
    */
    config.resize(2);
    config[0] = k;
    config[1] = 0;
    conf_ref = dev.global_argument(config);
    // TODO: release buffers no longer needed
    // we should reconfigure the NDRange of fuse_prep-actor here to k
    auto idx_ref = dev.scratch_argument<uval>(2 * k, buffer_type::output);
    self->send(fuse_prep, conf_ref, chid_ref, lits_ref, idx_ref);
    self->receive(
      [&](uref&, uref&, uref&, uref&) {
        //cout << "Prepared index." << endl;
      }
    );
    //auto idx_res = idx_ref.data();
    //valid_or_exit(idx_res, "Index preparation glitched.");
    // stream compaction using input ad valid
    // currently newly created actor to change NDRange
    wi = round_up(2 * k, 128u);
    auto index_space_2k_128 = spawn_config{dim_vec{wi}, {}, dim_vec{128}};
    sc_count = mngr.spawn_stage<uvec,uvec,uvec,uvec>(prog_sc, "countElts",
                                                 index_space_2k_128);
    sc_move = mngr.spawn_stage<uvec,uvec,uvec,uvec,
                               uvec,uvec,uvec,uvec>(prog_sc,
                                                "moveValidElementsStaged",
                                                index_space_2k_128);
    blocks_size = wi / 128; // wgs;
    blocks_ref = dev.scratch_argument<uval>(blocks_size, buffer_type::output);
    config.resize(2);
    config[0] = 2 * k;
    config[1] = 0;
    conf_ref = dev.global_argument(config);
    self->send(sc_count, conf_ref, blocks_ref, idx_ref, b128_ref);
    self->receive(
      [&](uref&, uref&, uref&, uref&) {
        //cout << "Count step done." << endl;
      }
    );
    out_ref = dev.scratch_argument<uval>(2 * k, buffer_type::output);
    self->send(sc_move, conf_ref,   idx_ref,  out_ref,  idx_ref,
                        blocks_ref, b128_ref, b128_ref, b128_ref);
    self->receive(
      [&](uref&, uref&, uref&, uref&,
          uref&, uref&, uref&, uref&) {
        //cout << "Merge step done." << endl;
      }
    );
    idx_ref.swap(out_ref);
    //cout << "DONE: fuse_fill_literals." << endl;
#ifdef SHOW_TIME_CONSUMPTION
    to = high_resolution_clock::now();
    cout << duration_cast<microseconds>(to - from).count()<< " us" << endl;
    from = high_resolution_clock::now();
#endif
    auto idx_conf = conf_ref.data();
    //valid_or_exit(res_conf, "Can't read conf after stream compaction.");
    auto conf = *idx_conf;
    auto idx_len = conf[1];
    //valid_or_exit(index_length == idx_len, "Lengths don't match");
 // cout << "Created index of length " << idx_len << "." << endl;
    auto idx_idx = idx_ref.data(idx_len);
    //valid_or_exit(idx_idx, "Can't read index after stream compaction.");
    auto idx = *idx_idx;
    // next step: compute_colum_length
    out_ref = dev.scratch_argument<uval>(k, buffer_type::output);
    auto heads_ref = dev.scratch_argument<uval>(k, buffer_type::output);
    auto index_space_k = spawn_config{dim_vec{k}};
    auto col_prep = mngr.spawn_stage<uvec,uvec,uvec,uvec>(prog_colum,
                                                      "colum_prepare",
                                                      index_space_k);
    auto col_scan = mngr.spawn_stage<uvec,uvec>(prog_colum,
                                              "lazy_segmented_scan",
                                              index_space_k);
    wi = round_up(k, 128u);
    auto index_space_k_128 = spawn_config{dim_vec{wi}, {}, dim_vec{128}};
    sc_count = mngr.spawn_stage<uvec,uvec,uvec,uvec>(prog_sc, "countElts",
                                                 index_space_k_128);
    sc_move = mngr.spawn_stage<uvec,uvec,uvec,uvec,
                               uvec,uvec,uvec,uvec>(prog_sc,
                                                "moveValidElementsStaged",
                                                index_space_k_128);
    // temp  -> the tmp array used by the algorithm
    // heads -> stores the heads array for the stream compaction
    self->send(col_prep, chid_ref, temp_ref, inpt_ref, heads_ref);
    self->receive(
      [&](uref&, uref&, uref&, uref&) {
        //cout << "Col: prepare done." << endl;
      }
    );
    self->send(col_scan, heads_ref, temp_ref);
    self->receive(
      [&](uref&, uref&) {
        //cout << "Col: scan done." << endl;
      }
    );
    config.resize(2);
    config[0] = k;
    config[1] = 0;
    conf_ref = dev.global_argument(config);
    self->send(sc_count, conf_ref, blocks_ref, heads_ref, b128_ref);
    self->receive(
      [&](uref&, uref&, uref&, uref&) {
        //cout << "Count step done." << endl;
      }
    );
    self->send(sc_move, conf_ref,   temp_ref, out_ref,  heads_ref,
                        blocks_ref, b128_ref, b128_ref, b128_ref);
    self->receive(
      [&](uref&, uref&, uref&, uref&,
          uref&, uref&, uref&, uref&) {
        //cout << "Merge step done." << endl;
      }
    );
    temp_ref.swap(out_ref);
    idx_conf = conf_ref.data();
    //valid_or_exit(res_conf, "Can't read conf after stream compaction.");
    conf = *idx_conf;
    auto keycount = conf[1];
    //valid_or_exit(keycnt == keycount, "Different amount of keys.");
 // cout << "Index has " << keycount << " keys." << endl;
    // missing: exclusive scan over temp_ref
    config.resize(2);
    config[0] = keycount;
    config[1] = 0;
    conf_ref = dev.global_argument(config);
    auto lazy_scan = mngr.spawn_stage<uvec,uvec,uvec>(prog_es, "lazy_scan",
                                                   spawn_config{dim_vec{1}});
    auto off_ref = dev.scratch_argument<uval>(k, buffer_type::output);
    self->send(lazy_scan, conf_ref, temp_ref, off_ref);
    self->receive(
      [&](uref&, uref&, uref&) {
        //cout << "Lazy scan done." << endl;
      }
    );
    auto offs_opt = off_ref.data(keycount);
/*
    auto scan_up = mngr.spawn_stage<uvec,uvec>(prog_es, "upsweep", index_space_k);
    auto scan_null = mngr.spawn_stage<uvec,uvec>(prog_es, "null_last",
                                               spawn_config{dim_vec{1}});
    auto scan_down = mngr.spawn_stage<uvec,uvec>(prog_es, "downsweep",
                                               index_space_k);
    uvec test_scan(k);
    std::iota(begin(test_scan), end(test_scan), 0u);
    temp_ref = dev.global_argument(test_scan);
    auto d = 0;
    config.resize(2);
    config[0] = k;
    config[1] = static_cast<uval>(d);
    conf_ref = dev.global_argument(config);
    self->send(scan_up, conf_ref, temp_ref);
    auto done = false;
    const auto bound = (std::log(k) / std::log(2)) - 1;
    self->receive_while([&] { return !done; })(
      [&](uref& conf, uref& temp) {
        d = d + 1;
        if (d <= bound) {
          config[1] = static_cast<uval>(d);
          conf = dev.global_argument(config);
          self->send(scan_up, conf, temp);
        } else {
          done = true;
        }
      }
    );
    cout << "Scan: upsweep done." << endl;
    self->send(scan_null, conf_ref, temp_ref);
    self->receive(
      [&](uref&, uref&) {
        cout << "Scan: null last done." << endl;
      }
    );
    d = bound; // (std::log(k) / std::log(2)) - 1;
    config.resize(2);
    config[0] = k;
    config[1] = static_cast<uval>(d);
    conf_ref = dev.global_argument(config);
    self->send(scan_down, conf_ref, temp_ref);
    done = false;
    self->receive_while([&] { return !done; })(
      [&](uref& conf, uref& temp) {
        d = d - 1;
        if (d >= 0) {
          config[1] = static_cast<uval>(d);
          conf = dev.global_argument(config);
          self->send(scan_down, conf, temp);
        } else {
          done = true;
        }
      }
    );
    cout << "Scan: upsweep done." << endl;
    auto offs_opt = temp_ref.data();
*/
    //valid_or_exit(offs_opt, "Can't read offsets back.");
    auto offs = *offs_opt;
    // idx_len --> length of index
    // keycount --> number of keys
    // idx --> contains index
    // offs --> contains offsets
    //valid_or_exit(offs == offsets, "Offsets differ.");
    auto stop = high_resolution_clock::now();
#ifdef SHOW_TIME_CONSUMPTION
    cout << duration_cast<microseconds>(stop - from).count()<< " us" << endl;
#endif
    cout //<< "Total: "
         << duration_cast<microseconds>(stop - start).count()
         << " us" << endl;
  }
  // clean up
  system.await_all_actors_done();
}

CAF_MAIN()
