
#include <cmath>
#include <tuple>
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
  template <>
  struct allowed_unsafe_message_type<opencl::dim_vec> : std::true_type {};
  template <>
  struct allowed_unsafe_message_type<spawn_config> : std::true_type {};
}

namespace {

using uval = uint32_t;
using uvec = std::vector<uval>;
using uref = mem_ref<uval>;

/*****************************************************************************\
                              JUST FOR STUFF
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

template <class T, class E = typename enable_if<is_integral<T>::value>::type>
T round_up(T numToRound, T multiple)  {
  assert(multiple > 0);
  return ((numToRound + multiple - 1) / multiple) * multiple;
}

template <class T>
vector<T> segmented_exclusive_scan(const vector<T>& data, const uvec& heads) {
  assert(data.size() == heads.size());
  assert(heads[0] == 1);
  vector<T> results(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    if (heads[i] == 1)
      results[i] = 0;
    else
      results[i] = results[i - 1] + data[i - 1];
  }
  return results;
}

template <class T>
vector<T> segmented_inclusive_scan(const vector<T>& data, const uvec& heads) {
  assert(data.size() == heads.size());
  assert(heads[0] == 1);
  vector<T> results(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    if (heads[i] == 1)
      results[i] = data[i];
    else
      results[i] = results[i - 1] + data[i];
  }
  return results;
}

/*****************************************************************************\
                          INTRODUCE SOME CLI ARGUMENTS
\*****************************************************************************/

class config : public actor_system_config {
public:
  size_t iterations = 1000;
  size_t threshold = 1500;
  string filename = "";
  uval bound = 0;
  string device_name = "GeForce GT 650M";
  bool print_results;
  config() {
    load<opencl::manager>();
    opt_group{custom_options_, "global"}
    .add(filename, "data-file,f", "file with test data (one value per line)")
    .add(bound, "bound,b", "maximum value (0 will scan values)")
    .add(device_name, "device,d", "device for computation (GeForce GTX 780M, "
                      "empty string will take first available device)")
    .add(print_results, "print,p", "print resulting bitmap index")
    .add(threshold, "threshold,t", "Threshold for output (1500)")
    .add(iterations, "iterations,i", "Number of times the empty kernel is supposed to run (1000)");
  }
};

} // namespace <anonymous>

/*****************************************************************************\
                                    MAIN!
\*****************************************************************************/

void caf_main(actor_system& system, const config& cfg) {
  uvec values;
  uvec heads;
  // ---- get data ----
  if (cfg.filename.empty()) {
    values.resize(16);
    iota(values.begin(), values.end(), 1);
    // cout << "A filename with test data is required, see --help for detailts."
    //      << endl;
    // return;
  } else {
    ifstream source{cfg.filename, std::ios::in};
    uval next;
    while (source >> next) {
      values.push_back(next);
    }
  }
  // ---- get device ----
  auto& mngr = system.opencl_manager();
  auto opt = mngr.get_device_if([&](const device& dev) {
      if (cfg.device_name.empty())
        return true;
      return dev.get_name() == cfg.device_name;
  });
  if (!opt) {
    opt = mngr.get_device_if([&](const device&) { return true; });
    if (!opt) {
      cout << "No device found." << endl;
      return;
    }
    cerr << "Using device '" << opt->get_name() << "'." << endl;
  }

  // --- scope to ensure actor cleanup ---
  {
    // ---- general ----
    auto dev = *opt;
    auto prog = mngr.create_program_from_file("./include/segmented_scan.cl",
                                              "", dev);
    // ---- input parameters ----
    size_t n = values.size();
    size_t group_size = 4;
    size_t global_range = round_up((n + 1) / 2, group_size);
    size_t local_range = group_size;
    /* // --- values used for direct block scan
    size_t n = values.size();
    size_t group_size = (n + 1) / 2;
    size_t global_range = round_up((n + 1) / 2, group_size);
    size_t local_range = group_size;
    */
    size_t groups = (global_range / local_range);
    heads.insert(begin(heads), n, 0);
    heads[0] = 1;
    heads[4] = 1;
    heads[8] = 1;
    heads[12] = 1;
    //heads[3] = 1;
    // ---- ndranges ----
    auto ndr_upsweep = spawn_config{dim_vec{global_range}, {},
                                    dim_vec{local_range}};
    //auto ndrange_g = spawn_config{dim_vec{round_up(groups, local_range)}, {},
    //                              dim_vec{local_range}};
    auto ndr_block = spawn_config{dim_vec{round_up(groups, local_range)}, {},
                                  dim_vec{local_range}};

    // ---- functions for arguments ----
    auto incs = [&](const uvec&, const uvec&, uval n) -> size_t {
      // calculate number of groups,
      // depending on the group size from the values size
      auto res = round_up((n + 1) / 2,
                          static_cast<uval>(group_size)) / group_size;
      //cout << "inc is " << res << endl;
      return res;
    };

    // ---- actors ----
    auto phase1 = mngr.spawn_new(prog, "upsweep", ndr_upsweep,
                                 in_out<uval, val, mref>{},    // data
                                 in_out<uval, val, mref>{},    // heads
                                 out<uval,mref>{incs},         // increments
                                 out<uval,mref>{incs},         // increment heads
                                 out<uval,mref>{incs},         // tree
                                 local<uval>{group_size * 2},  // data buffer
                                 local<uval>{group_size * 2},  // heads buffer
                                 priv<uval, val>{});
    auto phase2 = mngr.spawn_new(prog, "block_scan", ndr_block, // length / 2 work items
                                 in_out<uval,mref,mref>{},     // increments
                                 in_out<uval,mref,mref>{},     // increments heads
                                 in_out<uval,mref,mref>{},     // tree
                                 priv<uval, val>{});            // length
    /*
    auto phase3 = mngr.spawn_new(prog, "downsweep", ndrange_h,
                                 in_out<uval,mref,mref>{},     // data
                                 in<uval,mref>{},              // heads
                                 in<uval,mref>{},              // increments
                                 in<uval,mref>{},              // increment heads
                                 priv<uval, val>{});
    */

    // ---- test data ----
    // TODO: ...

    // ---- computations -----
    scoped_actor self{system};
    uref d, h;
    self->send(phase1, values, heads, static_cast<uval>(n));
    self->receive([&](uref& data, uref& heads, uref& incs, 
                      uref& inc_heads, uref& tree) {
      d = data;
      h = heads;
      self->send(phase2, incs, inc_heads, tree, static_cast<uval>(groups));
    });
    self->receive([&](uref& incs, uref& inc_heads, uref& tree) {
      //self->send(phase3, data, increments, n);
    });
    /*
    self->receive([&](const uvec& results) {
      
    });
    */
    /*
    self->send(phase2, values, heads, heads, static_cast<uval>(n));
    auto scanned = segmented_exclusive_scan(values, heads);
    self->receive([&](uvec& data, uvec&, uvec&) {
      if (data != scanned) {
        cout << "Expected different result" << endl;
        for (size_t i = 0; i < data.size(); ++i) {
          //if (scanned[i] != data[i]) {
            cout << "[" << i << "] " << scanned[i] << " : " << data[i] << endl;
          //}
        }
      } else {
        cout << "Success" << endl;
      }
    });
    */
  }

  // ---- DONE ----
  system.await_all_actors_done();
}

CAF_MAIN()
