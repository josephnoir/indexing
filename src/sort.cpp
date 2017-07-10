/******************************************************************************
 * Copyright (C) 2017                                                         *
 * Raphael Hiesgen <raphael.hiesgen (at) haw-hamburg.de>                      *
 *                                                                            *
 * Distributed under the terms and conditions of the BSD 3-Clause License.    *
 *                                                                            *
 * If you did not receive a copy of the license files, see                    *
 * http://opensource.org/licenses/BSD-3-Clause and                            *
 ******************************************************************************/

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

constexpr const char* kernel_file = "./include/radix.cl";

} // namespace anonymous

// required to allow sending mem_ref<int> in messages
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(uref);
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(radix_config);
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(opencl::dim_vec);
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(nd_range);

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

template <class T, typename std::enable_if<is_integral<T>{}, int>::type = 0>
uval as_uval(T val) { return static_cast<uval>(val); }

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
  string device_name = "AMD Radeon Pro 560 Compute Engine";
  bool print_results;
  config() {
    load<opencl::manager>();
    opt_group{custom_options_, "global"}
    .add(filename, "data-file,f", "file with test data (one value per line)")
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
  cout << "reading values from '" << cfg.filename << "' ..." << endl;
  ifstream source{cfg.filename, std::ios::in};
  uval next;
  while (source >> next)
    values.push_back(next);
  auto bound = cfg.bound;
  if (bound == 0 && !values.empty()) {
    auto itr = max_element(values.begin(), values.end());
    bound = *itr;
  }
  cout << "got '" << values.size() << "' values" << endl;

  auto& mngr = system.opencl_manager();

  // get device named in config ...
  auto opt = mngr.find_device_if([&](const device_ptr dev) {
    if (cfg.device_name.empty())
      return true;
    return dev->name() == cfg.device_name;
  });
  // ... or first one available
  if (!opt)
    opt = mngr.find_device_if([&](const device_ptr) { return true; });
  if (!opt) {
    cerr << "no device found" << endl;
    return;
  }
  cout << "using '" << (*opt)->name() << "' for computation" << endl;
  auto dev = *opt;

  // load kernels from source
  auto prog_radix  = mngr.create_program_from_file(kernel_file, "", dev);

  // configuration parameters
  auto n = values.size();
  // sort configuration
  // thread block has multiple thread groups has multiple threads
  // - blocks match to OpenCL work groups
  // - threads map to work items, but ids are counted inside a block
  // - groups separate threads of a block into multiple bundles
  uint32_t cardinality = 16;
  uint32_t l_val = 4; // bits used as a bucket in each radix iteration
  uint32_t radices = 1 << l_val;
  uint32_t blocks = 16;
    //= (dev->get_max_compute_units() <= (radices / 2)) ? (radices / 2) : radices;
  uint32_t threads_per_block = 256;
    //= max(radices, static_cast<uint32_t>(dev->get_max_work_group_size()));
  uint32_t threads_per_group = 16; //threads_per_block / 8; //radices;
  uint32_t groups_per_block = threads_per_block / threads_per_group;
  uint32_t mask = (1 << l_val) - 1;
  uint32_t radices_per_block = radices / blocks;
  uint32_t elements = static_cast<uint32_t>(n);
  uint32_t groups = groups_per_block * blocks;
  uint32_t elements_per_group = (elements / groups) + 1;
  // groups share a counter and each group requires a counter per radix
  uint32_t number_of_counters = radices * groups_per_block * blocks;
  uint32_t number_of_prefixes = radices;
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
  auto ndr_radix = nd_range{dim_vec{threads_per_block * blocks}, {},
                            dim_vec{threads_per_block}};
  {
    auto start = high_resolution_clock::now();
    // create phases
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
      prog_radix, "reorder", ndr_radix,
      in_out<uval,mref,mref>{},
      in_out<uval,mref,mref>{},
      in_out<uval,mref,mref>{},
      in_out<uval,mref,mref>{},
      local<uval>{groups_per_block * radices},
      local<uval>{radices},
      priv<radix_config>{rc},
      priv<uval,val>{}
    );
    
    scoped_actor self{system};
    auto start_sort = chrono::high_resolution_clock::now();
    // kernel executions
    uref input = dev->global_argument(values);
    {
      // radix sort for values by key using inpt as keys and temp as values
      auto keys_in = input;
      auto keys_out = dev->scratch_argument<uval>(n, buffer_type::input_output);
      auto counters = dev->scratch_argument<uval>(number_of_counters);
      auto prefixes = dev->scratch_argument<uval>(number_of_prefixes);
      uint32_t iterations = cardinality / l_val;
      for (uint32_t i = 0; i < iterations; ++i) {
        uval offset = l_val * i;
        self->send(radix_count, keys_in, counters, offset);
        self->receive([&](uref& k, uref& c) {
          self->send(radix_scan, k, c, prefixes, offset);
        });
        self->receive([&](uref& k, uref& c, uref& p) {
          self->send(radix_move, k, keys_out, c, p, offset);
        });
        self->receive([&](uref& i, uref& o, uref& c, uref& p) {
          std::swap(keys_in, o);
          std::swap(keys_out, i);
          counters = std::move(c);
          prefixes = std::move(p);
        });
      }
      input = keys_in;
    }
    auto eres = input.data();
    if (!eres) {
      cout << "error: " << system.render(eres.error()) << endl;
      return;
    }
    auto stop_sort = chrono::high_resolution_clock::now();
    cout << "radix sort:       " << right << setw(7)
         << duration_cast<microseconds>(stop_sort - start_sort).count()
         << " us" << endl;
    
    start_sort = chrono::high_resolution_clock::now();
    stable_sort(begin(values), end(values));
    stop_sort = chrono::high_resolution_clock::now();
    cout << "std::stable_sort: " << right << setw(7)
         << duration_cast<microseconds>(stop_sort - start_sort).count()
         << " us" << endl;

    auto stop = high_resolution_clock::now();
    cout << "total:            "  << right << setw(7)
         << duration_cast<microseconds>(stop - start).count() << " us" << endl;
    
    auto result = *eres;
    auto failures = 0;
    for (size_t i = 0; i < result.size(); ++i)
      if (result[i] != values[i])
        ++failures;
    cout << "radix sort failed for " << failures << " values" << endl;
  }
  // clean up
  system.await_all_actors_done();
}

CAF_MAIN()
