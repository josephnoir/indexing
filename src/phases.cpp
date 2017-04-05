
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

//#define WITH_CPU_TESTS
#define SHOW_TIME_CONSUMPTION
#define WITH_DESCRIPTION
#ifdef WITH_CPU_TESTS
# undef SHOW_TIME_CONSUMPTION
#endif // WITH_CPU_TESTS
#ifdef WITH_DESCRIPTION
# define DESCRIPTION(x) (x)
#else
# define DESCRIPTION(x) ""
#endif

#define POSITION string(__FILE__) + ":" + to_string(__LINE__)

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
  template <>
  struct allowed_unsafe_message_type<opencl::dim_vec> : std::true_type {};
  template <>
  struct allowed_unsafe_message_type<spawn_config> : std::true_type {};
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
                     TESTS FUNCTIONS ON CPU FOR COMPARISON
\*****************************************************************************/

#ifdef WITH_CPU_TESTS
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
      // Original code:
      //if (chids[i] != 0)
      //  new_chids[i] = chids[i] - 1;
      // ... which seems to result in the loss of fills
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
#endif // WITH_CPU_TESTS

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
    cout << "Using device named '" << opt->get_sizeame() << "'." << endl;
  }*/
  auto dev = *opt;

#ifdef WITH_CPU_TESTS
  // Create test data
#endif // WITH_CPU_TESTS

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

  // configuration parameters
  auto n = values.size();
  size_t max_wg_size = dev.get_max_work_group_size();
  auto ndrange = spawn_config{dim_vec{n}};
  auto ndrange_rounded = spawn_config{dim_vec{round_up(n, max_wg_size)}, {},
                                      dim_vec{max_wg_size}};
  auto wi = round_up(n, 128ul);
  auto ndrange_128  = spawn_config{dim_vec{wi}, {}, dim_vec{128}};
  // TODO: not a nice solution, need some better appraoch
  auto wi_two = [&](uvec&, uref&) { return size_t{wi / 128}; };
  auto k_once = [](uref&, uref&, uref&, uval k) { return size_t{k}; };
  auto one = [](uref&, uref&, uref&, uval) { return size_t{1}; };
  auto k_double = [](uref&, uref&, uval k) { return size_t{2 * k}; };
  auto k_two = [](uref&, uval k) { return size_t{k}; };
  // exclusive scan
  auto es_m = wi / 128;
  size_t es_group_size = 128;
  size_t es_global_range = round_up((es_m + 1) / 2, es_group_size);
  size_t es_local_range = es_group_size;
  size_t es_groups = (es_global_range / es_local_range);
  auto es_range_h = spawn_config{dim_vec{es_global_range}, {},
                                 dim_vec{es_local_range}};
  auto es_range_g = spawn_config{dim_vec{round_up(es_groups, es_local_range)},
                                 {}, dim_vec{es_local_range}};
  auto es_incs = [&](const uvec&, uval n) -> size_t {
    // calculate number of groups, depending on the group size from the input size
    return (round_up((n + 1) / 2, static_cast<uval>(es_group_size)) / es_group_size);
  };
  auto get_size = [](const uref& in) -> size_t { return in.size(); };


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
    // sort rids ...
    auto rids_1 = mngr.spawn_new(prog_rids, "create_rids", ndrange,
                                 in_out<uval,val,mref>{}, out<uval,mref>{},
                                 priv<uval>{static_cast<uval>(n)});
    // radix sort (by key)
    auto radix_zero = mngr.spawn_new(prog_radix, "zeroes", zero_range,
                                     in_out<uval,mref,mref>{});
    auto radix_count = mngr.spawn_new(prog_radix, "count", radix_range,
                                      in_out<uval,mref,mref>{},
                                      in_out<uval,mref,mref>{},
                                      priv<radix_config>{rc},
                                      priv<uval,val>{});
    auto radix_scan = mngr.spawn_new(prog_radix, "scan", radix_range,
                                     in_out<uval,mref,mref>{},
                                     in_out<uval,mref,mref>{},
                                     in_out<uval,mref,mref>{},
                                     local<uval>{groups_per_block * blocks},
                                     priv<radix_config>{rc},
                                     priv<uval,val>{});
    auto radix_move = mngr.spawn_new(prog_radix, "reorder_kv", radix_range,
                                     in_out<uval,mref,mref>{},
                                     in_out<uval,mref,mref>{},
                                     in_out<uval,mref,mref>{},
                                     in_out<uval,mref,mref>{},
                                     in_out<uval,mref,mref>{},
                                     in_out<uval,mref,mref>{},
                                     local<uval>{groups_per_block * radices},
                                     local<uval>{radices},
                                     priv<radix_config>{rc},
                                     priv<uval,val>{});
    // produce chuncks ...
    /*
    auto chunks = mngr.spawn_new(prog_chunks, "produce_chunks", ndrange,
                                 in_out<uval,mref,mref>{},
                                 out<uval,mref>{}, out<uval,mref>{});
    */
    auto chunks = mngr.spawn_new(prog_chunks, "produce_chunks2",
                                 ndrange_rounded,
                                 in_out<uval,mref,mref>{},
                                 out<uval,mref>{get_size},
                                 out<uval,mref>{get_size},
                                 priv<uval>{static_cast<uval>(n)});
    /*
    auto chunks = mngr.spawn_new(prog_chunks, "produce_chunks3",
                                 ndrange_rounded,
                                 in_out<uval,mref,mref>{},
                                 out<uval,mref>{get_size},
                                 out<uval,mref>{get_size},
                                 priv<uval>{static_cast<uval>(n)});
    */
    // <uvec, uvec, uvec>
    auto merge_heads = mngr.spawn_new(prog_merge, "create_heads", ndrange,
                                      in_out<uval,mref,mref>{},
                                      in_out<uval,mref,mref>{},
                                      out<uval,mref>{});
    auto merge_scan = mngr.spawn_new(prog_merge, "lazy_segmented_scan",
                                     ndrange, in<uval,mref>{},
                                     in_out<uval,mref,mref>{});
    // stream compaction
    auto sc_count = mngr.spawn_new(prog_sc,"countElts", ndrange_128,
                                   out<uval,mref>{wi_two},
                                   in_out<uval,mref,mref>{},
                                   local<uval>{128},
                                   priv<uval,val>{});
    // sum operation is handled by es actors belows (exclusive scan)
    auto sc_move = mngr.spawn_new(prog_sc, "moveValidElementsStaged",
                                  ndrange_128,
                                  out<uval,val>{one},
                                  in_out<uval,mref,mref>{},
                                  out<uval,mref>{k_once},
                                  in_out<uval,mref,mref>{},
                                  in_out<uval,mref,mref>{},
                                  local<uval>{128},
                                  local<uval>{128},
                                  local<uval>{128},
                                  priv<uval,val>{});
    // produce fills
    auto fills = mngr.spawn_new(prog_fills, "produce_fills", ndrange,
                                in<uval,mref>{},in<uval,mref>{},
                                out<uval,mref>{},priv<uval,val>{});
    // fuse fill & literals
    auto fuse_prep = mngr.spawn_new(prog_fuse, "prepare_index", ndrange,
                                    in<uval,mref>{},in<uval,mref>{},
                                    out<uval,mref>{k_double},
                                    priv<uval,val>{});
    // compute column length
    auto col_prep = mngr.spawn_new(prog_colum, "colum_prepare", ndrange,
                                   in_out<uval,mref,mref>{}, out<uval,mref>{},
                                   in_out<uval,mref,mref>{}, out<uval,mref>{});
    auto col_scan = mngr.spawn_new(prog_colum, "lazy_segmented_scan", ndrange,
                                   in_out<uval,mref,mref>{},
                                   in_out<uval,mref,mref>{});
    // scan
    auto lazy_scan = mngr.spawn_new(prog_es, "lazy_scan",
                                    spawn_config{dim_vec{1}},
                                    in<uval,mref>{},
                                    out<uval,mref>{k_two},
                                    priv<uval,val>{});
    // exclusive scan
    auto es1 = mngr.spawn_new(prog_es, "es_phase_1", es_range_h,
                              in_out<uval, mref, mref>{},
                              out<uval,mref>{es_incs},
                              local<uval>{es_group_size * 2},
                              priv<uval, val>{});
    auto es2 = mngr.spawn_new(prog_es, "es_phase_2", es_range_g,
                              in_out<uval,mref,mref>{},
                              in_out<uval,mref,mref>{},
                              local<uval>{es_group_size * 2},
                              priv<uval, val>{});
    auto es3 = mngr.spawn_new(prog_es, "es_phase_3", es_range_h,
                              in_out<uval,mref,mref>{},
                              in<uval,mref>{},
                              priv<uval, val>{});
#ifdef SHOW_TIME_CONSUMPTION
    auto to = high_resolution_clock::now();
    auto from = high_resolution_clock::now();
#endif

    // kernel executions
    scoped_actor self{system};
    self->send(rids_1, values);
    uref input_r;
    uref rids_r;
    self->receive([&](uref& in, uref& rids) {
      input_r = move(in);
      rids_r = move(rids);
    });

#ifdef WITH_CPU_TESTS
    {
      auto input_exp = input_r.data();
      auto input = *input_exp;
      valid_or_exit(input == values, POSITION);
    }
#endif // WITH_CPU_TESTS

    {
      // radix sort for values by key using inpt as keys and temp as values
      auto r_keys_in = input_r;
      auto r_values_in = rids_r;
      auto r_keys_out = dev.scratch_argument<uval>(n, buffer_type::input_output);
      auto r_values_out = dev.scratch_argument<uval>(n, buffer_type::input_output);
      // TODO: see how performance is affected if we create new arrays each time
      auto r_counters = dev.scratch_argument<uval>(counters);
      auto r_prefixes = dev.scratch_argument<uval>(prefixes);
      uint32_t iterations = cardinality / l_val;
      for (uint32_t i = 0; i < iterations; ++i) {
        uval offset = l_val * i;
        if (i > 0) {
          std::swap(r_keys_in, r_keys_out);
          std::swap(r_values_in, r_values_out);
        }
        self->send(radix_zero, r_counters);
        self->receive([&](uref& /*counters*/) { });
        self->send(radix_count, r_keys_in, r_counters, offset);
        self->receive([&](uref& /*k*/, uref& /*c*/) { });
        self->send(radix_scan, r_keys_in, r_counters, r_prefixes, offset);
        self->receive([&](uref& /*k*/, uref& /*c*/, uref& /*p*/) { });
        self->send(radix_move, r_keys_in, r_keys_out, r_values_in, r_values_out,
                               r_counters, r_prefixes, offset);
        self->receive([&](uref& /*ki*/, uref& /*ko*/, uref& /*vi*/, uref& /*vo*/,
                          uref& /*c*/, uref& /*p*/) { });
      }
      input_r = r_keys_out;
      rids_r = r_values_out;
    }

#ifdef WITH_CPU_TESTS
    auto test_input = values;
    uvec test_rids(test_input.size());
    uvec test_chids(test_input.size());
    uvec test_lits(test_input.size());
    sort_rids_by_value(test_input, test_rids);
    auto input_exp = input_r.data();
    auto rids_exp = rids_r.data();
    if (!input_exp) {
      cout << "Can't read keys back (" << system.render(input_exp.error())
           << ")." << endl;
    } else if (!rids_exp) {
      cout << "Can't read values back (" << system.render(rids_exp.error())
           << ")." << endl;
    } else {
      auto input = *input_exp;
      auto rids = *rids_exp;
      vector<size_t> failed_keys;
      vector<size_t> failed_values;
      for (size_t i = 0; i < input.size(); ++i) {
        if (input[i] != test_input[i])
          failed_keys.push_back(i);
        if (rids[i] != test_rids[i])
          failed_values.push_back(i);
      }
      if (!failed_keys.empty() || !failed_values.empty()) {
        cout << "failed_keys for "
             << failed_keys.size() << " keys and "
             << failed_values.size() << " values." << endl;
        return;
      }
    }
#endif // WITH_CPU_TESTS

#ifdef SHOW_TIME_CONSUMPTION
    dev.synchronize();
    //cout << "DONE: sort_rids_by_value" << endl;
    to = high_resolution_clock::now();
    cout << DESCRIPTION("sort_rids_by_value:\t\t")
         << duration_cast<microseconds>(to - from).count() << " us" << endl;
    from = high_resolution_clock::now();
#endif

    self->send(chunks, rids_r);
    uref chids_r;
    uref lits_r;
    self->receive([&](uref& /*rids*/, uref& chids, uref& lits) {
      chids_r = chids;
      lits_r = lits;
    });

#ifdef WITH_CPU_TESTS
    produce_chunk_id_literals(test_rids, test_chids, test_lits);
    auto in_exp = input_r.data();
    auto ch_exp = chids_r.data();
    auto li_exp = lits_r.data();
    valid_or_exit(in_exp, POSITION);
    valid_or_exit(ch_exp, POSITION);
    valid_or_exit(li_exp, POSITION);
    auto in = *in_exp;
    auto ch = *ch_exp;
    auto li = *li_exp;
    {
      vector<size_t> ch_failures;
      vector<size_t> li_failures;
      for (size_t i = 0; i < ch.size(); ++i) {
        if (ch[i] != test_chids[i])
          ch_failures.push_back(i);
        if (li[i] != test_lits[i])
          li_failures.push_back(i);
      }
      if (!ch_failures.empty() || !li_failures.empty()) {
        cout << "Failed for " << ch_failures.size() << " chids and "
             << li_failures.size() << " literals." << endl;
        return;
      }
    }
    valid_or_exit((in == test_input), POSITION);
    valid_or_exit((ch == test_chids), POSITION);
    valid_or_exit((li == test_lits), POSITION);
#endif // WITH_CPU_TESTS

#ifdef SHOW_TIME_CONSUMPTION
    dev.synchronize();
    //cout << "DONE: produce_chunk_id_literals" << endl;
    to = high_resolution_clock::now();
    cout << DESCRIPTION("produce_chunk_id_literals:\t")
         << duration_cast<microseconds>(to - from).count()<< " us" << endl;
    from = high_resolution_clock::now();
#endif

    uref heads_r;
    uval k = 0;
#ifdef SHOW_TIME_CONSUMPTION
    auto t1 = high_resolution_clock::now();
#endif
    self->send(merge_heads, input_r, chids_r);
    self->receive([&](uref&, uref&, uref& heads) {
      heads_r = heads;
    });
#ifdef SHOW_TIME_CONSUMPTION
    auto t2 = high_resolution_clock::now();
    dev.synchronize();
#endif
    // cout << "Created heads array" << endl;
    self->send(merge_scan, heads_r, lits_r);
    self->receive([&](uref&) { });
#ifdef SHOW_TIME_CONSUMPTION
    dev.synchronize();
    auto t3 = high_resolution_clock::now();
#endif
    // cout << "Merged values" << endl;
    // stream compact inpt, chid, lits by heads value
    uref blocks_r;
    uval len = static_cast<uval>(n);
    self->send(sc_count, heads_r, len);
    self->receive([&](uref& blocks, uref&) {
      blocks_r = blocks;
    });
#ifdef SHOW_TIME_CONSUMPTION
    dev.synchronize();
    auto t4 = high_resolution_clock::now();
#endif
    self->send(es1, blocks_r, static_cast<uval>(es_m));
    self->receive([&](uref& data, uref& increments) {
      self->send(es2, data, increments, static_cast<uval>(es_groups));
    });
    self->receive([&](uref& data, uref& increments) {
      self->send(es3, data, increments, static_cast<uval>(es_m));
    });
    self->receive([&](const uref& results) {
      blocks_r = results;
    });
#ifdef SHOW_TIME_CONSUMPTION
    dev.synchronize();
    auto t5 = high_resolution_clock::now();
#endif
    // cout << "Count step done." << endl;
    // TODO: Can we do this concurrently?
    self->send(sc_move, input_r, heads_r, blocks_r, len);
    self->receive([&](uvec& res, uref&, uref& out, uref&, uref&) {
      k = res[0];
      input_r.swap(out);
    });
#ifdef SHOW_TIME_CONSUMPTION
    auto t6 = high_resolution_clock::now();
#endif
    //cout << "Merge step done (input)." << endl;
    self->send(sc_move, chids_r, heads_r, blocks_r, len);
    self->receive([&](uvec& res, uref&, uref& out, uref&, uref&) {
      k = res[0];
      chids_r.swap(out);
    });
#ifdef SHOW_TIME_CONSUMPTION
    auto t7 = high_resolution_clock::now();
#endif
    // cout << "Merge step done (chids)." << endl;
    self->send(sc_move, lits_r, heads_r, blocks_r, len);
    self->receive([&](uvec& res, uref&, uref& out, uref&, uref&) {
      k = res[0];
      lits_r.swap(out);
    });
#ifdef SHOW_TIME_CONSUMPTION
    auto t8 = high_resolution_clock::now();
#endif
    // cout << "Merge step done (lits)." << endl;

#ifdef SHOW_TIME_CONSUMPTION
    // cout << "DONE: merge_lit_by_val_chids." << endl;
    to = high_resolution_clock::now();
    cout << DESCRIPTION("merge_lit_by_val_chids:\t\t")
         << duration_cast<microseconds>(to - from).count()<< " us" << endl;
    cout << " merge headers: " << duration_cast<microseconds>(t2 - t1).count() << endl
         << " merge scan   : " << duration_cast<microseconds>(t3 - t2).count() << endl
         << " sc count     : " << duration_cast<microseconds>(t4 - t3).count() << endl
         << " sc scan      : " << duration_cast<microseconds>(t5 - t4).count() << endl
         << " sc move      : " << duration_cast<microseconds>(t6 - t5).count() << endl
         << " sc move      : " << duration_cast<microseconds>(t7 - t6).count() << endl
         << " sc move      : " << duration_cast<microseconds>(t8 - t7).count() << endl;
    from = high_resolution_clock::now();
#endif

#ifdef WITH_CPU_TESTS
    auto test_k = merged_lit_by_val_chids(test_input, test_chids, test_lits);
    valid_or_exit(k == test_k, POSITION);
    auto res_inpt = input_r.data();
    auto res_chid = chids_r.data();
    auto res_lits = lits_r.data();
    valid_or_exit(res_inpt, POSITION);
    valid_or_exit(res_chid, POSITION);
    valid_or_exit(res_lits, POSITION);
    uvec new_inpt{*res_inpt};
    uvec new_chid{*res_chid};
    uvec new_lits{*res_lits};
    new_inpt.resize(k);
    new_chid.resize(k);
    new_lits.resize(k);
    valid_or_exit(new_inpt == test_input, POSITION + " input not equal");
    valid_or_exit(new_chid == test_chids, POSITION + " chids not equal");
    valid_or_exit(new_lits == test_lits, POSITION + " lits not equal");
#endif // WITH_CPU_TESTS


    self->send(fills, spawn_config{dim_vec{k}}, input_r, chids_r, k);
    self->receive([&](uref& out) { chids_r.swap(out); });

#ifdef WITH_CPU_TESTS
    uvec test_chids_produce{test_chids};
    produce_fills(test_input, test_chids_produce, test_k);
    res_inpt = input_r.data();
    res_chid = chids_r.data();
    valid_or_exit(res_inpt, POSITION + " destroyed input");
    valid_or_exit(res_chid, POSITION + " can't read fills");
    new_inpt = *res_inpt;
    new_chid = *res_chid;
    new_inpt.resize(k);
    new_chid.resize(k);
    valid_or_exit(new_inpt == test_input, "input not equal");
    valid_or_exit(new_chid == test_chids_produce, "chids not equal");
#endif // WITH_CPU_TESTS

#ifdef SHOW_TIME_CONSUMPTION
    dev.synchronize();
    // cout << "DONE: produce fills." << endl;
    to = high_resolution_clock::now();
    cout << DESCRIPTION("produce_fills:\t\t\t")
         << duration_cast<microseconds>(to - from).count()<< " us" << endl;
    from = high_resolution_clock::now();
#endif

    uref index_r;
    size_t index_length = 0;
#ifdef SHOW_TIME_CONSUMPTION
    auto t11 = high_resolution_clock::now();
#endif
    self->send(fuse_prep, spawn_config{dim_vec{k}}, chids_r, lits_r, k);
    self->receive([&](uref& index) { index_r = index; });
    auto idx_e = index_r.data();
    // cout << "Prepared index." << endl;
    wi = round_up(2 * k, 128u);
    auto ndrange_2k_128 = spawn_config{dim_vec{wi}, {}, dim_vec{128}};
    // new calculactions for scan
    es_m = wi / 128;
    es_global_range = round_up((es_m + 1) / 2, es_group_size);
    es_groups = (es_global_range / es_local_range);
    es_range_h = spawn_config{dim_vec{es_global_range}, {},
                              dim_vec{es_local_range}};
    es_range_g = spawn_config{dim_vec{round_up(es_groups, es_local_range)}, {},
                              dim_vec{es_local_range}};
#ifdef SHOW_TIME_CONSUMPTION
    dev.synchronize();
    auto t12 = high_resolution_clock::now();
#endif
    self->send(sc_count, ndrange_2k_128, index_r, 2 * k);
    self->receive([&](uref& blocks, uref&) {
      blocks_r = blocks;
    });
#ifdef SHOW_TIME_CONSUMPTION
    dev.synchronize();
    auto t13 = high_resolution_clock::now();
#endif
    self->send(es1, es_range_h, blocks_r, static_cast<uval>(es_m));
    self->receive([&](uref& data, uref& increments) {
      self->send(es2, es_range_g, data, increments, static_cast<uval>(es_groups));
    });
    self->receive([&](uref& data, uref& increments) {
      self->send(es3, es_range_h, data, increments, static_cast<uval>(es_m));
    });
    self->receive([&](const uref& results) {
      blocks_r = results;
    });
#ifdef SHOW_TIME_CONSUMPTION
    dev.synchronize();
    auto t14 = high_resolution_clock::now();
#endif
    self->send(sc_move, ndrange_2k_128, index_r, index_r, blocks_r, 2 * k);
    self->receive([&](uvec& res, uref&, uref& out, uref&, uref&) {
      index_length = res[0];
      index_r = out;
      //cout << "Merge step done." << endl;
    });
#ifdef SHOW_TIME_CONSUMPTION
    auto t15 = high_resolution_clock::now();
#endif

#ifdef SHOW_TIME_CONSUMPTION
    //cout << "DONE: fuse_fill_literals." << endl;
    to = high_resolution_clock::now();
    cout << DESCRIPTION("fuse_fill_literals:\t\t")
         << duration_cast<microseconds>(to - from).count()<< " us" << endl;
    cout << " fuse prep    : " << duration_cast<microseconds>(t12 - t11).count() << endl
         << " sc count     : " << duration_cast<microseconds>(t13 - t12).count() << endl
         << " sc scan      : " << duration_cast<microseconds>(t14 - t13).count() << endl
         << " sc move      : " << duration_cast<microseconds>(t15 - t14).count() << endl;
    from = high_resolution_clock::now();
#endif

    // next step: compute_colum_length
    auto ndrange_k = spawn_config{dim_vec{k}};
    // temp  -> the tmp array used by the algorithm
    // heads -> stores the heads array for the stream compaction
    uref tmp_r;
    self->send(col_prep, ndrange_k, chids_r, input_r);
    self->receive([&](uref& chids, uref& tmp, uref& input, uref& heads) {
      chids_r = chids;
      tmp_r = tmp;
      input_r = input;
      heads_r = heads;
    });
    // cout << "Col: prepare done." << endl;
    self->send(col_scan, ndrange_k, heads_r, tmp_r);
    self->receive([&](uref& heads, uref& tmp) {
      heads_r = heads;
      tmp_r = tmp;
    });
    // cout << "Col: scan done." << endl;

    wi = round_up(k, 128u);
    uval keycount = 0;
    auto ndrange_k_128 = spawn_config{dim_vec{wi}, {}, dim_vec{128}};
    auto out_ref = dev.scratch_argument<uval>(k, buffer_type::output);
    // new calculations for scan actors
    es_m = wi / 128;
    es_global_range = round_up((es_m + 1) / 2, es_group_size);
    es_groups = (es_global_range / es_local_range);
    es_range_h = spawn_config{dim_vec{es_global_range}, {},
                              dim_vec{es_local_range}};
    es_range_g = spawn_config{dim_vec{round_up(es_groups, es_local_range)}, {},
                              dim_vec{es_local_range}};
    // stream compaction
    self->send(sc_count, ndrange_k_128, heads_r, k);
    self->receive([&](uref& blocks, uref& heads) {
      blocks_r = blocks;
      heads_r = heads;
    });
    // cout << "Count step done." << endl;
    self->send(es1, es_range_h, blocks_r, static_cast<uval>(es_m));
    self->receive([&](uref& data, uref& increments) {
      self->send(es2, es_range_g, data, increments, static_cast<uval>(es_groups));
    });
    self->receive([&](uref& data, uref& increments) {
      self->send(es3, es_range_h, data, increments, static_cast<uval>(es_m));
    });
    self->receive([&](const uref& results) {
      blocks_r = results;
    });
    self->send(sc_move, ndrange_k_128, tmp_r, heads_r, blocks_r, k);
    self->receive([&](uvec& data, uref&, uref& out, uref&, uref&) {
      keycount = data[0];
      heads_r = out;
    });
    //cout << "Merge step done." << endl;

    // TODO: better exclusive scan over heads_r
    uref offsets_r;
    self->send(lazy_scan, heads_r, keycount);
    self->receive([&](uref& offsets) {
      offsets_r = offsets;
    });
    //cout << "Lazy scan done." << endl;

#ifdef SHOW_TIME_CONSUMPTION
    dev.synchronize();
    to = high_resolution_clock::now();
    cout << DESCRIPTION("compute_colum_length:\t\t")
         << duration_cast<microseconds>(to - from).count() << " us" << endl;
    from = high_resolution_clock::now();
#endif

    auto index_opt = index_r.data(index_length);
    auto offsets_opt = offsets_r.data(keycount);

#ifdef SHOW_TIME_CONSUMPTION
    to = high_resolution_clock::now();
    cout << DESCRIPTION("Reading back data:\t\t")
         << duration_cast<microseconds>(to - from).count()<< " us" << endl;
    from = high_resolution_clock::now();
#endif

#ifdef WITH_CPU_TESTS
    valid_or_exit(index_opt, "Can't read index.");
    valid_or_exit(offsets_opt, "Can't read offsets.");
    auto index = *index_opt;
    auto offsets = *offsets_opt;
    // create test index
    uvec test_index(2 * test_k);
    uvec test_chids_fuse{test_chids_produce};
    auto test_index_length = fuse_fill_literals(test_chids_fuse, test_lits,
                                                test_index, test_k);
    valid_or_exit(test_index_length == index_length, "Index lengths don't match");
    valid_or_exit(index == test_index, "Indexes differ.");
    // create test offsets
    uvec test_offsets(test_k);
    uvec test_input_col{test_input};
    auto test_keycount = compute_colum_length(test_input_col, test_chids_fuse,
                                              test_offsets, test_k);
    valid_or_exit(test_keycount == keycount, "Offsets have different keycount.");
    valid_or_exit(offsets == test_offsets, "Offsets differ.");
    cout << "Run included tests of calculated data." << endl
         << "Test index has " << test_index_length << " elements with "
         << test_keycount << " keys." << endl
         << "Program got " << index_length << " elements with "
         << keycount << " keys." << endl;
#endif // WITH_CPU_TESTS

    auto stop = high_resolution_clock::now();
    cout << DESCRIPTION("Total:\t\t\t\t")
         << duration_cast<microseconds>(stop - start).count() << " us" << endl;
    // Calculated data:
    // index_length --> length of index
    // keycount     --> number of keys
    // index        --> contains index
    // offsets      --> contains offsets
  }
  // clean up
  system.await_all_actors_done();
}

CAF_MAIN()
