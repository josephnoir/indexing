
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

using uval = uint32_t;
using uvec = std::vector<uval>;
using uref = mem_ref<uval>;

using add_atom = atom_constant<atom("add")>;
using init_atom = atom_constant<atom("init")>;
using quit_atom = atom_constant<atom("quit")>;
using index_atom = atom_constant<atom("index")>;

constexpr const char* kernel_file_09 = "./include/empty_kernel.cl";

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

/*****************************************************************************\
                          INTRODUCE SOME CLI ARGUMENTS
\*****************************************************************************/

class config : public actor_system_config {
public:
  size_t iterations = 1000;
  size_t threshold = 1500;
  string filename = "";
  uval bound = 0;
  string device_name = "";//"GeForce GTX 780M";
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

  // load kernels
  auto prog_overhead = mngr.create_program_from_file(kernel_file_09, "", dev);

  // create spawn configuration
  size_t n = 1;
  auto index_space_overhead = spawn_config{dim_vec{n}};
  // buffers for execution
  uvec config{static_cast<uval>(n)};
  {
    // create phases
    auto overhead = mngr.spawn_stage<uint*>(prog_overhead, "empty_kernel", index_space_overhead);
    // kernel executions
    // temp_ref used as rids buffer
    scoped_actor self{system};
    auto conf_ref = dev.global_argument(config);

    auto start = high_resolution_clock::now();
    auto stop = start;
    vector<size_t> measurements(cfg.iterations);
    for(size_t i = 0; i < cfg.iterations; ++i) {
      start = high_resolution_clock::now();
      self->send(overhead, conf_ref);
      self->receive([&](uref&) {
        stop = high_resolution_clock::now();
      });
      measurements[i] = duration_cast<microseconds>(stop - start).count();
    }

    auto amount = 0;
    auto threshold = cfg.threshold;
    auto brk = 0;
    for (size_t i = 0; i < measurements.size(); ++i) {
      if (measurements[i] > threshold) {
        cout << setw(4) << i << " (" << setw(5) << measurements[i]  << ")   "; // << endl;
        amount += 1;
        ++brk;
        if (brk > 13) {
          cout << endl;
          brk = 0;
        }
      }
    }
    if (brk != 0)
      cout << endl;
    cout << amount << " of " << cfg.iterations << " values were above " << threshold << endl;
    //auto stop = high_resolution_clock::now();
    // TODO check if microseconds are good enough or if we should use nanoseconds instead
    /*
    cout << "Time: '"
         << duration_cast<microseconds>(stop - start).count()
         << "' us" << endl;
    */
  }
  // clean up
  system.await_all_actors_done();
}

CAF_MAIN()
