
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
//  {
//    auto prog = mngr.create_program_from_file("./include/scan.cl", "", dev);
//    size_t n = 1024;
//    size_t group_size = 128;
//    size_t global_range = round_up((n + 1) / 2, group_size);
//    size_t local_range = group_size;
//    size_t groups = (global_range / local_range);
//    // test data
//    uvec input(n);
//    std::iota(begin(input), end(input), 1);
//    // spawn arguments
//    auto ndrange_h = spawn_config{dim_vec{global_range}, {}, dim_vec{local_range}};
//    auto ndrange_g = spawn_config{dim_vec{round_up(groups, local_range)}, {},
//                                  dim_vec{local_range}};
//    auto incs = [&](const uvec&, uval n) -> size_t {
//      // calculate number of groups, depending on the group size from the input size
//      return (round_up((n + 1) / 2, static_cast<uval>(group_size)) / group_size);
//    };
//    // actors
//    auto es1 = mngr.spawn_new(prog, "es_phase_1", ndrange_h,
//                              in_out<uval, val, mref>{},
//                              out<uval,mref>{incs},
//                              local<uval>{group_size * 2},
//                              priv<uval, val>{});
//    auto es2 = mngr.spawn_new(prog, "es_phase_2", ndrange_g,
//                              in_out<uval,mref,mref>{},
//                              in_out<uval,mref,mref>{},
//                              local<uval>{group_size * 2},
//                              priv<uval, val>{});
//    auto es3 = mngr.spawn_new(prog, "es_phase_3", ndrange_h,
//                              in_out<uval,mref,val>{},
//                              in<uval,mref>{},
//                              priv<uval, val>{});
//    // computations
//    scoped_actor self{system};
//    self->send(es1, input, static_cast<uval>(n));
//    self->receive([&](uref& data, uref& increments) {
//      self->send(es2, data, increments, static_cast<uval>(groups));
//    });
//    self->receive([&](uref& data, uref& increments) {
//      self->send(es3, data, increments, static_cast<uval>(n));
//    });
//    self->receive([&](const uvec& results) {
//      uval curr = 0;
//      bool error = false;
//      for (size_t i = 0; i < results.size(); ++i) {
//        if (results[i] != curr) {
//          cout << input[i] << " --> " << results[i]
//               << " ***** should be " << curr << endl;
//          error = true;
//        }
//        curr += input[i];
//      }
//      if (!error)
//        cout << "Success!" << endl;
//    });
//  }
  {
    auto prog_comp = mngr.create_program_from_file("./include/stream_compaction.cl","",dev);
    auto prog_scan = mngr.create_program_from_file("./include/scan.cl", "", dev);
    size_t n = 1024 * 10;
    auto wi = round_up(n, 128ul);
    auto es_m = wi / 128;
    size_t es_group_size = 128;
    size_t es_global_range = round_up((es_m + 1) / 2, es_group_size);
    size_t es_local_range = es_group_size;
    size_t es_groups = (es_global_range / es_local_range);
    // test data
    uvec input(n);
    std::iota(begin(input), end(input), 1);
    auto is_prime = [](uval num) {
      if (num <= 3) {
        return num > 1;
      } else if (num % 2 == 0 || num % 3 == 0) {
        return false;
      } else {
        for (uval i = 5; i * i <= num; i += 6) {
          if (num % i == 0 || num % (i + 2) == 0) {
            return false;
          }
        }
        return true;
      }
    };
    uvec heads(input.size());
    std::transform(begin(input), end(input), begin(heads), [&](uval& elem) {
      return is_prime(elem) ? 1 : 0;
    });
    // spawn arguments
    auto ndrange_128  = spawn_config{dim_vec{wi}, {}, dim_vec{128}};
    // TODO: not a nice solution, need some better appraoch
    auto wi_two = [&](uvec&, uval& n) { return size_t{round_up(n, 128u) / 128u}; };
    auto k_once = [](uref&, uref&, uref&, uval k) { return size_t{k}; };
    auto one = [](uref&, uref&, uref&, uval) { return size_t{1}; };
    auto es_range_h = spawn_config{dim_vec{es_global_range}, {}, dim_vec{es_local_range}};
    auto es_range_g = spawn_config{dim_vec{round_up(es_groups, es_local_range)}, {},
                                  dim_vec{es_local_range}};
    auto es_incs = [&](const uvec&, uval n) -> size_t {
      // calculate number of groups, depending on the group size from the input size
      return (round_up((n + 1) / 2, static_cast<uval>(es_group_size)) / es_group_size);
    };
    auto sc_count = mngr.spawn_new(prog_comp,"countElts", ndrange_128,
                                   out<uval,mref>{wi_two},
                                   in_out<uval,val,mref>{},
                                   local<uval>{128},
                                   priv<uval,val>{});
    auto es1 = mngr.spawn_new(prog_scan, "es_phase_1", es_range_h,
                              in_out<uval, mref, mref>{},
                              out<uval,mref>{es_incs},
                              local<uval>{es_group_size * 2},
                              priv<uval, val>{});
    auto es2 = mngr.spawn_new(prog_scan, "es_phase_2", es_range_g,
                              in_out<uval,mref,mref>{},
                              in_out<uval,mref,mref>{},
                              local<uval>{es_group_size * 2},
                              priv<uval, val>{});
    auto es3 = mngr.spawn_new(prog_scan, "es_phase_3", es_range_h,
                              in_out<uval,mref,mref>{},
                              in<uval,mref>{},
                              priv<uval, val>{});
    auto sc_move = mngr.spawn_new(prog_comp, "moveValidElementsStaged",
                                  ndrange_128,
                                  out<uval,val>{one},
                                  in<uval,val>{},
                                  out<uval,mref>{k_once},
                                  in<uval,mref>{},
                                  in<uval,mref>{},
                                  local<uval>{128},
                                  local<uval>{128},
                                  local<uval>{128},
                                  priv<uval,val>{});
    // computations
    scoped_actor self{system};
    self->send(sc_count, heads, static_cast<uval>(n));
    uref blocks;
    uref valid;
    self->receive([&](uref& b, uref& v) {
      blocks = b;
      valid = v;
    });
    self->send(es1, blocks, static_cast<uval>(es_m));
    self->receive([&](uref& data, uref& increments) {
      self->send(es2, data, increments, static_cast<uval>(es_groups));
    });
    self->receive([&](uref& data, uref& increments) {
      self->send(es3, data, increments, static_cast<uval>(es_m));
    });
    self->receive([&](const uref& results) {
      blocks = results;
    });
    self->send(sc_move, input, valid, blocks, static_cast<uval>(n));
    self->receive([&](uvec& size, uref& data) {
      auto vec = data.data(size.front());
      vec = data.data(size.front());
      vec = data.data(size.front());
      cout << "Created compacted array of " << size.front() << " values." << endl;
      for (auto e : *vec)
        cout << setw(6) << e << " ";
      cout << endl;
    });
  }
  // clean up
  system.await_all_actors_done();
}

CAF_MAIN()
