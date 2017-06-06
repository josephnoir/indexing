
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
using upair = std::pair<uval,uval>;

using add_atom = atom_constant<atom("add")>;
using init_atom = atom_constant<atom("init")>;
using quit_atom = atom_constant<atom("quit")>;
using index_atom = atom_constant<atom("index")>;

constexpr const char* kernel_file_01 = "./include/sort_rids_by_value.cl";
constexpr const char* kernel_file_02 = "./include/produce_chunk_id_literals.cl";
constexpr const char* kernel_file_03 = "./include/merge_lit_by_val_chids.cl";
constexpr const char* kernel_file_04 = "./include/produce_fills.cl";
constexpr const char* kernel_file_05 = "./include/fuse_fill_literals.cl";
constexpr const char* kernel_file_06 = "./include/compute_column_length.cl";
constexpr const char* kernel_file_07 = "./include/stream_compaction.cl";
constexpr const char* kernel_file_08 = "./include/scan.cl";
constexpr const char* kernel_file_09 = "./include/radix.cl";
constexpr const char* kernel_file_10 = "./include/segmented_scan.cl";

//constexpr const char* kernel_name_01a = "kernel_wah_index";

} // namespace anonymous

// required to allow sending mem_ref<int> in messages
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(uref);
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(radix_config);
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(opencl::dim_vec);
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(spawn_config);

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
  // auto tmp = res.size();
  // s.str(string());
  // s << tmp;
  //return s.str();
  res.erase(res.find_last_not_of("0") + 1);
  return res;
}
*/

template <class T, typename std::enable_if<is_integral<T>{}, int>::type = 0>
uval as_uval(T val) { return static_cast<uval>(val); }

/*****************************************************************************\
                     TESTS FUNCTIONS ON CPU FOR COMPARISON
\*****************************************************************************/

#ifdef WITH_CPU_TESTS

// in : input, rids (rids will be resized and filled with the rids)
// out: input, rids (both sorted by input)
void sort_rids_by_value(vector<uval>& input, vector<uval>& rids) {
  auto size = input.size();
  rids.resize(size);
  iota(begin(rids), end(rids), 0);
  vector<upair> view;
  view.reserve(size);
  for (size_t i = 0; i < size; ++i)
    view.emplace_back(input[i], rids[i]);
  std::stable_sort(begin(view), end(view),
    [](const upair& lhs, const upair& rhs) {
      return lhs.first < rhs.first;
    }
  );
  for (size_t i = 0; i < view.size(); ++i) {
    input[i] = view[i].first;
    rids[i] = view[i].second;
  }
}

// in : rids, n (length)
// out: chids, lits
void produce_chunk_id_literals(vector<uval>& rids,
                                vector<uval>& chids,
                                vector<uval>& lits) {
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
size_t merged_lit_by_val_chids(vector<uval>& input,
                               vector<uval>& chids,
                               vector<uval>& lits) {
  return reduce_by_key(input, chids, lits);
}

// in : input, chids, k (reduced length)
// out: chids with 0-fill symbols
void produce_fills(vector<uval>& input,
                   vector<uval>& chids,
                   size_t k) {
  vector<uval> heads(k);
  adjacent_difference(begin(input), begin(input) + k, begin(heads));
  heads.front() = 1;
  vector<uval> new_chids(k);
  for (size_t i = 0; i < k; ++i) {
    if (heads[i] == 0) {
      new_chids[i] = chids[i] - chids[i - 1] - 1;
    } else {
      new_chids[i] = chids[i];
      // --- Original code ---
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
size_t fuse_fill_literals(vector<uval>& chids, vector<uval>& lits,
                          vector<uval>& index, size_t k) {
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
size_t compute_column_length(vector<uval>& input,
                             vector<uval>& chids,
                             vector<uval>& offsets,
                             size_t k) {
  vector<uval> tmp(k);
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
  string device_name = "GeForce GT 650M";
  bool print_results;
  config() {
    load<opencl::manager>();
    opt_group{custom_options_, "global"}
    .add(filename, "data-file,f", "file with test data (one value per line)")
    .add(bound, "bound,b", "maximum value (0 will scan values)")
    .add(device_name, "device,d", "device for computation (GeForce GT 650M, "
                      ", but will take first available device if not found)")
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
  ifstream source{cfg.filename, std::ios::in};
  uval next;
  while (source >> next)
    values.push_back(next);
  auto bound = cfg.bound;
  if (bound == 0 && !values.empty()) {
    auto itr = max_element(values.begin(), values.end());
    bound = *itr;
  }

  auto& mngr = system.opencl_manager();

  // get device named in config ...
  auto opt = mngr.get_device_if([&](const device_ptr dev) {
    if (cfg.device_name.empty())
      return true;
    return dev->get_name() == cfg.device_name;
  });
  // ... or first one available
  if (!opt)
    opt = mngr.get_device_if([&](const device_ptr) { return true; });
  if (!opt) {
    cerr << "No device found." << endl;
    return;
  }
  cout << "Using device '" << (*opt)->get_name() << "'." << endl;
  auto dev = *opt;

#ifdef WITH_CPU_TESTS
  // Create test data
#endif // WITH_CPU_TESTS

  // load kernels from source
  auto prog_rids   = mngr.create_program_from_file(kernel_file_01, "", dev);
  auto prog_chunks = mngr.create_program_from_file(kernel_file_02, "", dev);
  auto prog_merge  = mngr.create_program_from_file(kernel_file_03, "", dev);
  auto prog_fills  = mngr.create_program_from_file(kernel_file_04, "", dev);
  auto prog_fuse   = mngr.create_program_from_file(kernel_file_05, "", dev);
  auto prog_column = mngr.create_program_from_file(kernel_file_06, "", dev);
  auto prog_sc     = mngr.create_program_from_file(kernel_file_07, "", dev);
  auto prog_es     = mngr.create_program_from_file(kernel_file_08, "", dev);
  auto prog_radix  = mngr.create_program_from_file(kernel_file_09, "", dev);
  auto prog_sscan  = mngr.create_program_from_file(kernel_file_10, "", dev);

  // configuration parameters
  auto n = values.size();
  auto ndr = spawn_config{dim_vec{n}};
  auto one = [](uref&, uref&, uref&, uval) { return size_t{1}; };
  auto k_double = [](uref&, uref&, uval k) { return size_t{2 * k}; };
  auto fills_k = [](uref&, uref&, uval k) { return size_t{k}; };
  // segmented scan
  auto half_block = dev->get_max_work_group_size() / 2;
  auto get_size = [half_block](size_t n) -> size_t {
    return round_up((n + 1) / 2, half_block);
  };
  auto half_size_for = [](size_t n, size_t block) -> size_t {
    return round_up((n + 1) / 2, block);
  };
  auto ndr_scan = [half_size_for, half_block](size_t dim) {
    return spawn_config{dim_vec{half_size_for(dim,half_block)}, {},
                                dim_vec{half_block}};
  };
  auto ndr_compact = [](uval dim) {
    return spawn_config{dim_vec{round_up(dim, 128u)}, {}, dim_vec{128}};
  };
  auto ndr_block = [half_size_for](uval n, uval block) {
    return spawn_config{dim_vec{half_size_for(n, block)}, {}, dim_vec{block}};
  };
  auto reduced_scan = [&](const uref&, uval n) {
    // calculate number of groups from the group size from the values size
    return size_t{get_size(n) / half_block};
  };
  auto reduced_sscan = [&](const uref&, const uref&, const uref&, uval n) {
    // calculate number of groups from the group size from the values size
    return size_t{get_size(n) / half_block};
  };
  auto reduced_compact = [](const uref&, uval n) {
    return size_t{round_up(n, 128u) / 128u};
  };
  auto k_compact = [](uref&, uref&, uref&, uval k) { return size_t{k}; };
  auto same_size = [&](const uref&, uval n) { return size_t{n}; };
  // sort configuration
  // thread block has multiple thread groups has multiple threads
  // - blocks match to OpenCL work groups
  // - threads map to work items, but ids are counted inside a block
  // - groups separate threads of a block into multiple bundles
  // TODO: Optimized runs with regard to cardinality
  uint32_t cardinality = 16;
  uint32_t l_val = 4; // bits used as a bucket in each radix iteration
  uint32_t radices = 1 << l_val;
  uint32_t blocks = 8;
    //= (dev->get_max_compute_units() <= (radices / 2)) ? (radices / 2) : radices;
  uint32_t threads_per_block = 1024;
    //= max(radices, static_cast<uint32_t>(dev->get_max_work_group_size()));
  uint32_t threads_per_group = 32; //threads_per_block / 8; //radices;
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
  auto ndr_radix = spawn_config{dim_vec{threads_per_block * blocks}, {},
                                dim_vec{threads_per_block}};
  {
    auto start = high_resolution_clock::now();
    // create phases
    // sort rids ...
    auto rids_1 = mngr.spawn(
      prog_rids, "create_rids", ndr,
      in_out<uval,mref,mref>{}, out<uval,mref>{},
      priv<uval>{as_uval(n)}
    );
    // ---- radix sort (by key) ----
    auto radix_count = mngr.spawn(
      prog_radix, "count", ndr_radix,
      in_out<uval,mref,mref>{},
      in_out<uval,mref,mref>{},
      local<uval>{radices * groups_per_block},
      priv<radix_config>{rc},
      priv<uval,val>{}
    );
    auto radix_scan = mngr.spawn(
      prog_radix, "scan", ndr_radix,
      in_out<uval,mref,mref>{},
      in_out<uval,mref,mref>{},
      in_out<uval,mref,mref>{},
      local<uval>{groups_per_block * blocks},
      priv<radix_config>{rc},
      priv<uval,val>{}
    );
    auto radix_move = mngr.spawn(
      prog_radix, "reorder_kv", ndr_radix,
      in_out<uval,mref,mref>{},
      in_out<uval,mref,mref>{},
      in_out<uval,mref,mref>{},
      in_out<uval,mref,mref>{},
      in_out<uval,mref,mref>{},
      in_out<uval,mref,mref>{},
      local<uval>{groups_per_block * radices},
      local<uval>{radices},
      priv<radix_config>{rc},
      priv<uval,val>{}
    );
    // ---- produce chuncks ----
    auto chunks = mngr.spawn(
      prog_chunks, "produce_chunks", ndr,
      in_out<uval,mref,mref>{},
      out<uval,mref>{}, out<uval,mref>{}
    );
    auto merge_heads = mngr.spawn(
      prog_merge, "create_heads", ndr,
      in_out<uval,mref,mref>{},
      in_out<uval,mref,mref>{},
      out<uval,mref>{}
    );
    auto merge_scan = mngr.spawn(
      prog_merge, "lazy_segmented_scan", ndr,
      in<uval,mref>{},
      in_out<uval,mref,mref>{}
    );
    // stream compaction
    auto sc_count = mngr.spawn(
      prog_sc,"countElts", ndr,
      [ndr_compact](spawn_config& conf, message& msg) -> optional<message> {
        msg.apply([&](const uref&, uval n) { conf = ndr_compact(n); });
        return std::move(msg);
      },
      out<uval,mref>{reduced_compact},
      in_out<uval,mref,mref>{},
      local<uval>{128},
      priv<uval,val>{}
    );
    // --> sum operation is handled by es actors belows (exclusive scan)
    auto sc_move = mngr.spawn(
      prog_sc, "moveValidElementsStaged", ndr,
      [ndr_compact](spawn_config& conf, message& msg) -> optional<message> {
        msg.apply([&](const uref&, const uref&, const uref&, uval n) {
          conf = ndr_compact(n);
        });
        return std::move(msg);
      },
      out<uval,val>{one},
      in_out<uval,mref,mref>{},
      out<uval,mref>{k_compact},
      in_out<uval,mref,mref>{},
      in_out<uval,mref,mref>{},
      local<uval>{128},
      local<uval>{128},
      local<uval>{128},
      priv<uval,val>{}
    );
    // ---- produce fills -----
    auto fills = mngr.spawn(
      prog_fills, "produce_fills", ndr,
      [](spawn_config& conf, message& msg) -> optional<message> {
        msg.apply([&](const uref&, const uref&, uval k) {
          conf = spawn_config{dim_vec{(k + 1) / 2}};
        });
        return move(msg);
      },
      in<uval,mref>{},in<uval,mref>{},
      out<uval,mref>{fills_k},
      priv<uval,val>{}
    );
    // ---- fuse fill & literals ----
    auto fuse_prep = mngr.spawn(
      prog_fuse, "prepare_index", ndr,
      [](spawn_config& conf, message& msg) -> optional<message> {
        msg.apply([&](const uref&, const uref&, uval k) {
          conf = spawn_config{dim_vec{k}};
        });
        return move(msg);
      },
      in<uval,mref>{},in<uval,mref>{},
      out<uval,mref>{k_double},
      priv<uval,val>{}
    );
    // compute column length
    auto col_prep = mngr.spawn(
      prog_column, "column_prepare", ndr,
      [ndr_block](spawn_config& conf, message& msg) -> optional<message> {
        msg.apply([&](const uref&, const uref&, uval n) {
          conf = ndr_block(n, 512);
        });
        return std::move(msg);
      },
      in_out<uval,mref,mref>{},
      in_out<uval,mref,mref>{},
      out<uval,mref>{},
      out<uval,mref>{},
      priv<uval,val>{}
    );
    auto col_conv = mngr.spawn(
      prog_column, "convert_heads", ndr,
      [ndr_scan](spawn_config& conf, message& msg) -> optional<message> {
        msg.apply([&](const uref&, uval n) { conf = ndr_scan(n); });
        return std::move(msg);
      },
      in<uval,mref>{},
      out<uval,mref>{same_size},
      priv<uval,val>{}
    );
    // exclusive scan
    auto scan1 = mngr.spawn(
      prog_es, "es_phase_1", ndr,
      [ndr_scan](spawn_config& conf, message& msg) -> optional<message> {
        msg.apply([&](const uref&, uval n) { conf = ndr_scan(n); });
        return std::move(msg);
      },
      in_out<uval, mref, mref>{},
      out<uval,mref>{reduced_scan},
      local<uval>{half_block * 2},
      priv<uval, val>{}
    );
    auto scan2 = mngr.spawn(
      prog_es, "es_phase_2",
      spawn_config{dim_vec{half_block}, {}, dim_vec{half_block}},
      in_out<uval,mref,mref>{},
      in_out<uval,mref,mref>{},
      priv<uval, val>{}
    );
    auto scan3 = mngr.spawn(
      prog_es, "es_phase_3", ndr,
      [ndr_scan](spawn_config& conf, message& msg) -> optional<message> {
        msg.apply([&](const uref&, const uref&, uval n) {
          conf = ndr_scan(n);
        });
        return std::move(msg);
      },
      in_out<uval,mref,mref>{},
      in<uval,mref>{},
      priv<uval, val>{}
    );
    // config for multi-level segmented scan
    auto seg_scan1 = mngr.spawn(
      prog_sscan, "upsweep", ndr,
      [ndr_scan](spawn_config& conf, message& msg) -> optional<message> {
        msg.apply([&](const uref&, const uref&, const uref&, uval n) {
          conf = ndr_scan(n);
        });
        return std::move(msg);
      },
      in_out<uval,mref,mref>{},       // data
      in_out<uval,mref,mref>{},       // partition
      in_out<uval,mref,mref>{},       // tree
      out<uval,mref>{reduced_sscan},  // last_data
      out<uval,mref>{reduced_sscan},  // last_part
      out<uval,mref>{reduced_sscan},  // last_tree
      local<uval>{half_block * 2},    // data buffer
      local<uval>{half_block * 2},    // heads buffer
      priv<uval, val>{}
    );
    auto seg_scan2 = mngr.spawn(
      prog_sscan, "block_scan",
      spawn_config{dim_vec{half_block}, {},
                   dim_vec{half_block}},
      in_out<uval,mref,mref>{},             // data
      in_out<uval,mref,mref>{},             // partition
      in<uval,mref>{},                      // tree
      priv<uval, val>{}                     // length
    );
    auto seg_scan3 = mngr.spawn(
      prog_sscan, "downsweep", ndr,
      [ndr_scan](spawn_config& conf, message& msg) -> optional<message> {
        msg.apply([&](const uref&, const uref&, const uref&, const uref&,
                      const uref&, uval n) {
          conf = ndr_scan(n);
        });
        return std::move(msg);
      },
      in_out<uval,mref,mref>{},     // data
      in_out<uval,mref,mref>{},     // partition
      in<uval,mref>{},              // tree
      in<uval,mref>{},              // last_data
      in<uval,mref>{},              // last_partition
      local<uval>{half_block * 2},  // data buffer
      local<uval>{half_block * 2},  // part buffer
      local<uval>{half_block * 2},  // tree buffer
      priv<uval, val>{}
    );
    auto seg_scan4 = mngr.spawn(
      prog_sscan, "downsweep_inc", ndr,
      [ndr_scan](spawn_config& conf, message& msg) -> optional<message> {
        msg.apply([&](const uref&, const uref&, const uref&, const uref&,
                      const uref&, const uref&, uval n) {
          conf = ndr_scan(n);
        });
        return std::move(msg);
      },
      in_out<uval,mref,mref>{},     // data
      in_out<uval,mref,mref>{},     // partition
      in<uval,mref>{},              // tree
      in<uval,mref>{},              // last_data
      in<uval,mref>{},              // last_partition
      in<uval,mref>{},              // original data
      local<uval>{half_block * 2},  // data buffer
      local<uval>{half_block * 2},  // part buffer
      local<uval>{half_block * 2},  // tree buffer
      priv<uval, val>{}
    );
    scoped_actor self{system};
#ifdef SHOW_TIME_CONSUMPTION
    auto to = high_resolution_clock::now();
    auto from = high_resolution_clock::now();
#endif
    // kernel executions
    uref input_r = dev->global_argument(values);
#ifdef SHOW_TIME_CONSUMPTION
    dev->synchronize();
    to = high_resolution_clock::now();
    cout << DESCRIPTION("Transfer to:\t\t")
         << duration_cast<microseconds>(to - from).count() << " us" << endl;
    from = high_resolution_clock::now();
#endif
    self->send(rids_1, input_r);
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
      auto r_keys_out = dev->scratch_argument<uval>(n, buffer_type::input_output);
      auto r_values_out = dev->scratch_argument<uval>(n, buffer_type::input_output);
      // TODO: see how performance is affected if we create new arrays each time
      auto r_counters = dev->scratch_argument<uval>(counters);
      auto r_prefixes = dev->scratch_argument<uval>(prefixes);
      uint32_t iterations = cardinality / l_val;
      vector<size_t> count_t(iterations);
      vector<size_t> scan_t(iterations);
      vector<size_t> move_t(iterations);
      auto start = high_resolution_clock::now();
      auto stop = high_resolution_clock::now();
      for (uint32_t i = 0; i < iterations; ++i) {
        uval offset = l_val * i;
        if (i > 0) {
          std::swap(r_keys_in, r_keys_out);
          std::swap(r_values_in, r_values_out);
        }
        start = high_resolution_clock::now();
        self->send(radix_count, r_keys_in, r_counters, offset);
        self->receive([&](uref& /*k*/, uref& /*c*/) {
          dev->synchronize();
          stop = high_resolution_clock::now();
          count_t[i] = duration_cast<microseconds>(stop - start).count();
          start = high_resolution_clock::now();
          self->send(radix_scan, r_keys_in, r_counters, r_prefixes, offset);
        });
        self->receive([&](uref& /*k*/, uref& /*c*/, uref& /*p*/) {
          dev->synchronize();
          stop = high_resolution_clock::now();
          scan_t[i] = duration_cast<microseconds>(stop - start).count();
          start = high_resolution_clock::now();
          self->send(radix_move, r_keys_in, r_keys_out, r_values_in, r_values_out,
                                 r_counters, r_prefixes, offset);
        });
        self->receive([&](uref& /*ki*/, uref& /*ko*/, uref& /*vi*/, uref& /*vo*/,
                          uref& /*c*/, uref& /*p*/) { });
        dev->synchronize();
        stop = high_resolution_clock::now();
        move_t[i] = duration_cast<microseconds>(stop - start).count();
      }
      input_r = r_keys_out;
      rids_r = r_values_out;
      size_t total_t = 0;
      total_t = 0;
      cout << "count: ";
      for (auto t : count_t){
        cout << setw(10) << t << "us";
        total_t += t;
      }
      cout << " = " << total_t  << "us" << endl;
      total_t = 0;
      cout << "scan:  ";
      for (auto t : scan_t){
        cout << setw(10) << t << "us";
        total_t += t;
      }
      cout << " = " << total_t  << "us" << endl;
      total_t = 0;
      cout << "move:  ";
      for (auto t : move_t){
        cout << setw(10) << t << "us";
        total_t += t;
      }
      cout << " = " << total_t  << "us" << endl;
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
        cout << "Sort failed for "
             << failed_keys.size() << " keys and "
             << failed_values.size() << " values." << endl;
        return;
      }
    }
#endif // WITH_CPU_TESTS

#ifdef SHOW_TIME_CONSUMPTION
    dev->synchronize();
    //cout << "DONE: sort_rids_by_value" << endl;
    to = high_resolution_clock::now();
    cout << DESCRIPTION("Sort:\t\t\t")
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
    dev->synchronize();
    //cout << "DONE: produce_chunk_id_literals" << endl;
    to = high_resolution_clock::now();
    cout << DESCRIPTION("Chunks + literals:\t")
         << duration_cast<microseconds>(to - from).count()<< " us" << endl;
    from = high_resolution_clock::now();
#endif

    uref heads_r;
    uval k = 0;
    self->send(merge_heads, input_r, chids_r);
    self->receive([&](uref&, uref&, uref& heads) {
      heads_r = heads;
    });
    // cout << "Created heads array" << endl;
    self->send(merge_scan, heads_r, lits_r);
    self->receive([&](uref&) { });
    //cout << "Merged values" << endl;
    // stream compact inpt, chid, lits by heads value
    uref blocks_r;
    uval len = as_uval(n);
    self->send(sc_count, heads_r, len);
    self->receive([&](uref& blocks, uref&) {
      self->send(scan1, blocks, as_uval(blocks.size()));
    });
    self->receive([&](uref& data, uref& incs) {
      self->send(scan2, data, incs, as_uval(incs.size()));
    });
    self->receive([&](uref& data, uref& incs) {
      self->send(scan3, data, incs, as_uval(data.size()));
    });
    self->receive([&](uref& results) {
      blocks_r = std::move(results);
    });
    //cout << "Count step done." << endl;
    // TODO: Can we do this concurrently?
    self->send(sc_move, input_r, heads_r, blocks_r, len);
    self->receive([&](uvec& res, uref&, uref& out, uref&, uref&) {
      k = res[0];
      std::swap(input_r, out);
    });
    //cout << "Merge step done (input)." << endl;
    self->send(sc_move, chids_r, heads_r, blocks_r, len);
    self->receive([&](uvec& res, uref&, uref& out, uref&, uref&) {
      k = res[0];
      std::swap(chids_r, out);
    });
    // cout << "Merge step done (chids)." << endl;
    self->send(sc_move, lits_r, heads_r, blocks_r, len);
    self->receive([&](uvec& res, uref&, uref& out, uref&, uref&) {
      k = res[0];
      std::swap(lits_r, out);
    });
    // cout << "Merge step done (lits)." << endl;

#ifdef SHOW_TIME_CONSUMPTION
    // cout << "DONE: merge_lit_by_val_chids." << endl;
    to = high_resolution_clock::now();
    cout << DESCRIPTION("Merge literals:\t\t")
         << duration_cast<microseconds>(to - from).count()<< " us" << endl;
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

    //self->send(fills, spawn_config{dim_vec{(k + 1) / 2}}, input_r, chids_r, k);
    self->send(fills, input_r, chids_r, k);
    self->receive([&](uref& out) { std::swap(chids_r, out); });

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
    dev->synchronize();
    // cout << "DONE: produce fills." << endl;
    to = high_resolution_clock::now();
    cout << DESCRIPTION("Produce fills:\t\t")
         << duration_cast<microseconds>(to - from).count()<< " us" << endl;
    from = high_resolution_clock::now();
#endif

    uref index_r;
    size_t index_length = 0;
    // self->send(fuse_prep, spawn_config{dim_vec{k}}, chids_r, lits_r, k);
    self->send(fuse_prep, chids_r, lits_r, k);
    self->receive([&](uref& index) { index_r = index; });
    // new calculactions for scan
    self->send(sc_count, index_r, 2 * k);
    self->receive([&](uref& blocks, uref&) {
      self->send(scan1, blocks, as_uval(blocks.size()));
    });
    self->receive([&](uref& data, uref& increments) {
      self->send(scan2, data, increments, as_uval(increments.size()));
    });
    self->receive([&](uref& data, uref& increments) {
      self->send(scan3, data, increments, as_uval(data.size()));
    });
    self->receive([&](const uref& results) {
      self->send(sc_move, index_r, index_r, results, 2 * k);
    });
    self->receive([&](uvec& res, uref&, uref& out, uref&, uref&) {
      index_length = res[0];
      index_r = out;
      //cout << "Merge step done." << endl;
    });

#ifdef SHOW_TIME_CONSUMPTION
    dev->synchronize();
    //cout << "DONE: fuse_fill_literals." << endl;
    to = high_resolution_clock::now();
    cout << DESCRIPTION("Fuse:\t\t\t")
         << duration_cast<microseconds>(to - from).count()<< " us" << endl;
    from = high_resolution_clock::now();
#endif

    // next step: compute_column_length
    // temp  -> the tmp array used by the algorithm
    // heads -> stores the heads array for the stream compaction
    uref tmp_r;
    auto tc1 = high_resolution_clock::now();
    self->send(col_prep, chids_r, input_r, as_uval(k));
    self->receive([&](uref& chids, uref& input, uref& tmp, uref& heads) {
      chids_r = move(chids);
      input_r = move(input);
      tmp_r = move(tmp);
      heads_r = move(heads);
    });
    dev->synchronize();
    auto tc2 = high_resolution_clock::now();

    // cout << "Col: prepare done." << endl;
    // --- segmented scan ---
    auto values_copy = dev->copy(tmp_r);
    auto heads_copy = dev->copy(heads_r);
    uref d, p, t, d2, p2, t2;
    self->send(seg_scan1, tmp_r, heads_r, *heads_copy,
               as_uval(tmp_r.size()));
    self->receive([&](uref&      data, uref&      part, uref&      tree,
                      uref& last_data, uref& last_part, uref& last_tree) {
      d = data;
      p = part;
      t = tree;
      self->send(seg_scan1, last_data, last_part, last_tree,
                 as_uval(last_data.size()));
    });
    self->receive([&](uref&      data, uref&      part, uref&      tree,
                      uref& last_data, uref& last_part, uref& last_tree) {
      d2 = data;
      p2 = part;
      t2 = tree;
      self->send(seg_scan2, last_data, last_part, last_tree,
                 as_uval(last_data.size()));
    });
    self->receive([&](uref& ld, uref& lp) {
      self->send(seg_scan3, d2, p2, t2, ld, lp, as_uval(d2.size()));
    });
    self->receive([&](uref& ld, uref& lp) {
      self->send(seg_scan4, d, p, t, ld, lp, std::move(*values_copy),
                 as_uval(d.size()));
    });
    self->receive([&](uref& results, const uref& /* partitions */) {
      tmp_r = std::move(results);
      self->send(col_conv, std::move(*heads_copy), k);
    });
    dev->synchronize();
    auto tc4 = high_resolution_clock::now();
    self->receive([&](const uref& new_heads) {
      heads_r = new_heads;
    });
    dev->synchronize();
    auto tc5 = high_resolution_clock::now();
    // cout << "Col: scan done." << endl;

    uval keycount = 0;
    // stream compaction
    self->send(sc_count, heads_r, k);
    self->receive([&](uref& blocks, uref& heads) {
      dev->synchronize();
      self->send(scan1, blocks, k);
      heads_r = heads;
    });
    // cout << "Count step done." << endl;
    self->receive([&](uref& data, uref& incs) {
      self->send(scan2, data, incs, as_uval(incs.size()));
    });
    self->receive([&](uref& data, uref& incs) {
      self->send(scan3, data, incs, k);
    });
    self->receive([&](const uref& results) {
      self->send(sc_move, tmp_r, heads_r, results, k);
    });
    self->receive([&](uvec& count, uref&, uref& out, uref&, uref&) {
      keycount = count[0];
      heads_r = out;
    });
    //cout << "Merge step done." << endl;
    dev->synchronize();
    auto tc6 = high_resolution_clock::now();

    uref offsets_r;

    self->send(scan1, heads_r, keycount);
    self->receive([&](uref& data, uref& incs) {
      d = std::move(data);
      self->send(scan1, incs, as_uval(incs.size()));
    });
    self->receive([&](uref& data, uref& incs) {
      self->send(scan2, data, incs, as_uval(incs.size()));
    });
    self->receive([&](uref& data, uref& incs) {
      self->send(scan3, data, incs, as_uval(data.size()));
    });
    self->receive([&](uref& incs) {
      self->send(scan3, d, incs, as_uval(d.size()));
    });
    self->receive([&](uref& results) {
      offsets_r = std::move(results);
    });
    dev->synchronize();
    auto tc7 = high_resolution_clock::now();

#ifdef SHOW_TIME_CONSUMPTION
    dev->synchronize();
    to = high_resolution_clock::now();
    cout << DESCRIPTION("Column length:\t\t")
         << duration_cast<microseconds>(to - from).count() << " us" << endl;
    cout << " > preparations:   " << duration_cast<microseconds>(tc2 - tc1).count() << " us" << endl;
    /*
    cout << " > segmented scan: " << duration_cast<microseconds>(tc3 - tc2).count() << " us" << endl;
    cout << " > make inclusive: " << duration_cast<microseconds>(tc4 - tc3).count() << " us" << endl;
    */
    cout << " > segmented scan: " << duration_cast<microseconds>(tc4 - tc2).count() << " us" << endl;
    cout << " > convert heads:  " << duration_cast<microseconds>(tc5 - tc4).count() << " us" << endl;
    cout << " > compaction:     " << duration_cast<microseconds>(tc6 - tc5).count() << " us" << endl;
    cout << " > scan:           " << duration_cast<microseconds>(tc7 - tc6).count() << " us" << endl;
    from = high_resolution_clock::now();
#endif

    auto index_opt = index_r.data(index_length);
    auto offsets_opt = offsets_r.data(keycount);

#ifdef SHOW_TIME_CONSUMPTION
    to = high_resolution_clock::now();
    cout << DESCRIPTION("Transfer back:\t\t")
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
    valid_or_exit(test_index_length == index_length,
                  "Index lengths don't match");
    valid_or_exit(index == test_index, "Indexes differ.");
    // create test offsets
    uvec test_offsets(test_k);
    uvec test_input_col{test_input};
    auto test_keycount = compute_column_length(test_input_col, test_chids_fuse,
                                               test_offsets, test_k);
    valid_or_exit(test_keycount == keycount,
                  "Offsets have different keycount.");
    valid_or_exit(offsets == test_offsets, "Offsets differ.");
    cout << "Run included tests of calculated data." << endl
         << "Test index has " << test_index_length << " elements with "
         << test_keycount << " keys." << endl
         << "Program got " << index_length << " elements with "
         << keycount << " keys." << endl;
#endif // WITH_CPU_TESTS

    auto stop = high_resolution_clock::now();
    cout << DESCRIPTION("Total:\t\t\t")
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
