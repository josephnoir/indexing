
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

// required to allow sending mem_ref<int> in messages
namespace caf {
  template <>
  struct allowed_unsafe_message_type<mem_ref<uint32_t>> : std::true_type {};
  template <>
  struct allowed_unsafe_message_type<mem_ref<radix_config>>
    : std::true_type {};
}

namespace {

using vec = std::vector<uint32_t>;
using val = vec::value_type;

constexpr const char* kernel_file_09 = "./include/radix.cl";

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
  val bits = 4;
  val blocks = 16;
  val threads = 0;
  val cardinality = 16;
  val group_size = 64;
  int loops = 1;
  string device_name = "Tesla C2075";
  bool print_results;
  config() {
    load<opencl::manager>();
    opt_group{custom_options_, "global"}
    .add(filename, "data-file,f", "file with test data (one value per line)")
    .add(device_name, "device,d", "choost device (empty string for first one)")
    .add(print_results, "print,p", "print resulting bitmap index")
    .add(bits, "bits,B", "Bits per iteration (default: 4)")
    .add(blocks, "blocks,b", "block for computation (radices / 2 or radices) ")
    .add(threads, "threads,t", "threads per block")
    .add(cardinality, "cardinality,c", "cardinality (default 16bit)")
    .add(group_size, "group_size,g", "threads per group (64)");
  }
};

} // namespace <anonymous>

/*****************************************************************************\
                                    MAIN!
\*****************************************************************************/

void caf_main(actor_system& system, const config& cfg) {
  vec keys;
  if (cfg.filename.empty()) {
    cout << "A filename with test data is required, see --help for detailts."
         << endl;
    return;
  }
  //cout << "Reading data from '" << cfg.filename << "' ... " << flush;
  ifstream source{cfg.filename, std::ios::in};
  val next;
  while (source >> next) {
    keys.push_back(next);
  }
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
  }
  auto dev = *opt;
  
  // load kernels
  auto prog_radix  = mngr.create_program_from_file(kernel_file_09, "", dev);

  // Create some help data
  auto n = keys.size();
  vec values(n);
  iota(begin(values), end(values), 0);

  // buffers we alread know we need
  
  auto keys_ref = dev.global_argument(keys);
  auto values_ref = dev.global_argument(values);
  auto keys_out_ref = dev.scratch_argument<val>(n, buffer_type::output);
  auto values_out_ref = dev.scratch_argument<val>(n, buffer_type::output);

  // sort configuration
  // thread block has multiple thread groups has multiple threads
  // - blocks match to OpenCL work groups
  // - threads map to work items, but ids are counted inside a block
  // - groups separate threads of a block into multiple bundles
  // TODO: Optimized runs with regard to cardinality
  uint32_t cardinality = cfg.cardinality;
  uint32_t l_val = cfg.bits;
  uint32_t radices = 1 << l_val;
  uint32_t blocks = cfg.blocks;
  if (blocks == 0)
    blocks = (dev.get_max_compute_units() <= (radices / 2)) ? (radices / 2) : radices;
  uint32_t threads_per_block = cfg.threads;
  if (threads_per_block == 0)
    threads_per_block = max(radices, static_cast<uint32_t>(dev.get_max_work_group_size()));
  uint32_t threads_per_group = cfg.group_size;
  if (threads_per_group == 0)
    threads_per_group = threads_per_block / radices;
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
  {
    scoped_actor self{system};
    // sort ndrange
    size_t radix_global = threads_per_block * blocks;
    size_t radix_local = threads_per_block;
    auto radix_range = spawn_config{dim_vec{radix_global}, {},
                                    dim_vec{radix_local}};
    auto zero_range = spawn_config{dim_vec{counters}};
    // sort actors
    auto radix_zero = mngr.spawn_phase<vec>(prog_radix, "zeroes", zero_range);
    auto radix_count = mngr.spawn_phase<vec,vec,radix_config,
                                        val>(prog_radix, "count", radix_range);
    auto radix_sum = mngr.spawn_phase<vec,vec,vec,vec,radix_config,
                                      val>(prog_radix, "scan", radix_range);
    auto radix_move
      = mngr.spawn_phase<vec,vec,vec,vec,vec,vec,vec,vec,
                         radix_config, val>(prog_radix, "reorder_kv",
                                              radix_range);
    // sort arguments
    auto r_keys_in = keys_ref;
    auto r_keys_out = keys_out_ref;
    auto r_values_in = values_ref;
    auto r_values_out = values_out_ref;
    auto r_counters = dev.scratch_argument<val>(counters);
    auto r_prefixes = dev.scratch_argument<val>(prefixes);
    auto r_local_a = dev.local_argument<val>(groups_per_block * blocks);
    auto r_local_b = dev.local_argument<val>(groups_per_block * radices);
    auto r_local_c = dev.local_argument<val>(radices);
    auto r_conf = dev.private_argument(rc);
    uint32_t iterations = cardinality / l_val;
    // sort itself
    auto from = high_resolution_clock::now();
    for (uint32_t i = 0; i < iterations; ++i) {
      auto r_offset = dev.private_argument(static_cast<val>(l_val * i));
      self->send(radix_zero, r_counters);
      self->receive([&](mem_ref<val>&) { });
      if (i > 0) {
        std::swap(r_keys_in, r_keys_out);
        std::swap(r_values_in, r_values_out);
      }
      self->send(radix_count, r_keys_in, r_counters, r_conf, r_offset);
      self->receive([&](mem_ref<val>&, mem_ref<val>&, mem_ref<radix_config>&,
                        mem_ref<val>&) { });
      self->send(radix_sum, r_keys_in, r_counters, r_prefixes, r_local_a,
                            r_conf, r_offset);
      self->receive([&](mem_ref<val>&, mem_ref<val>&, mem_ref<val>&,
                        mem_ref<val>&,
                        mem_ref<radix_config>&, mem_ref<val>&) { });
      self->send(radix_move, r_keys_in, r_keys_out,
                             r_values_in, r_values_out,
                             r_counters, r_prefixes,
                             r_local_b, r_local_c,
                             r_conf, r_offset);
      self->receive([&](mem_ref<val>&, mem_ref<val>&, mem_ref<val>&,
                        mem_ref<val>&, mem_ref<val>&, mem_ref<val>&,
                        mem_ref<val>&, mem_ref<val>&,
                        mem_ref<radix_config>&, mem_ref<val>&) { });
    }
    auto to = high_resolution_clock::now();
    // cleanup
    keys_ref = r_keys_out;
    values_ref = r_values_out;
    // time
    //cout << duration_cast<microseconds>(to - from).count() << " us" << endl;
    /*
    // Create test data
    auto input = keys;
    vec rids(input.size());
    vec chids(input.size());
    vec lits(input.size());
    sort_rids_by_value(input, rids);
    // check results
    auto keys_expected = keys_ref.data();
    auto values_expected = values_ref.data();
    if (!keys_expected) {
      cout << "Can't read keys back ("
           << system.render(keys_expected.error())
           << ")." << endl;
    } else if (!values_expected) {
      cout << "Can't read values back ("
           << system.render(values_expected.error())
           << ")." << endl;
    } else {
      auto inp = *keys_expected;
      auto rid = *values_expected;
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
    */
  }
  // done
  system.await_all_actors_done();
}

CAF_MAIN()
