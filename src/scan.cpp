
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
  {
    auto prog = mngr.create_program_from_file("./include/scan.cl", "", dev);
    size_t n = 1024;
    size_t group_size = 128;
    size_t global_range = round_up((n + 1) / 2, group_size);
    size_t local_range = group_size;
    size_t groups = (global_range / local_range);
    // test data
    uvec input(n);
    std::iota(begin(input), end(input), 1);
    // spawn arguments
    auto ndrange_h = spawn_config{dim_vec{global_range}, {}, dim_vec{local_range}};
    auto ndrange_g = spawn_config{dim_vec{round_up(groups, local_range)}, {},
                                  dim_vec{local_range}};
    auto incs = [&](const uvec&, uval n) -> size_t {
      // calculate number of groups, depending on the group size from the input size
      return (round_up((n + 1) / 2, static_cast<uval>(group_size)) / group_size);
    };
    // actors
    auto es1 = mngr.spawn_new(prog, "es_phase_1", ndrange_h,
                              in_out<uval, val, mref>{},
                              out<uval,mref>{incs},
                              local<uval>{group_size * 2},
                              priv<uval, val>{});
    auto es2 = mngr.spawn_new(prog, "es_phase_2", ndrange_g,
                              in_out<uval,mref,mref>{},
                              in_out<uval,mref,mref>{},
                              local<uval>{group_size * 2},
                              priv<uval, val>{});
    auto es3 = mngr.spawn_new(prog, "es_phase_3", ndrange_h,
                              in_out<uval,mref,val>{},
                              in<uval,mref>{},
                              priv<uval, val>{});
    // computations
    scoped_actor self{system};
    self->send(es1, input, static_cast<uval>(n));
    self->receive([&](uref& data, uref& increments) {
      self->send(es2, data, increments, static_cast<uval>(groups));
    });
//    self->receive([&](uvec& data, uvec& increments) {
//      for (size_t i = 0; i < data.size(); ++i)
//        cout << input[i] << " --> " << data[i] << endl;
//      for (auto e : increments)
//        cout << e << " ";
//      cout << endl;
//    });
    self->receive([&](uref& data, uref& increments) {
      self->send(es3, data, increments, static_cast<uval>(n));
    });
    self->receive([&](const uvec& results) {
      uval curr = 0;
      bool error = false;
      for (size_t i = 0; i < results.size(); ++i) {
        if (results[i] != curr) {
          cout << input[i] << " --> " << results[i]
               << " ***** should be " << curr << endl;
          error = true;
        }
        curr += input[i];
      }
      if (!error)
        cout << "Success!" << endl;
    });
  }
//  {
//    auto is_prime = [](uval num) {
//      if (num <= 3) {
//        return num > 1;
//      } else if (num % 2 == 0 || num % 3 == 0) {
//        return false;
//      } else {
//        for (uval i = 5; i * i <= num; i += 6) {
//          if (num % i == 0 || num % (i + 2) == 0) {
//            return false;
//          }
//        }
//        return true;
//      }
//    };
//    uvec values(70324);
//    iota(begin(values), end(values), 1);
//    uvec heads(values.size());
//    std::transform(begin(values), end(values), begin(heads), [&](uval& elem) {
//      return is_prime(elem) ? 1 : 0;
//    });
//    auto n = values.size();
//    auto wi = round_up(n, 128ul);
//    auto m = wi / 128;
//    auto gr = round_up(m, 128ul);
//    auto ndrange_128  = spawn_config{dim_vec{wi}, {}, dim_vec{128}};
//    auto ndrange_128_sum = spawn_config{dim_vec{gr}, {}, dim_vec{128}};
//    // TODO: not a nice solution, need some better appraoch
//    auto wi_one = [](uref&, uval n) { return size_t{(n / 128) + 1u}; };
//    auto wi_two = [&](uvec&, uval& n) { return size_t{round_up(n, 128u) / 128u}; };
//    auto k_once = [](uref&, uref&, uref&, uval k) { return size_t{k}; };
//    auto one = [](uref&, uref&, uref&, uval) { return size_t{1}; };
//    auto prog_sc = mngr.create_program_from_file("./include/stream_compaction.cl","",dev);
//    auto sc_count = mngr.spawn_new(prog_sc,"countElts", ndrange_128,
//                                   out<uval,mref>{wi_two},
//                                   in_out<uval,val,mref>{},
//                                   local<uval>{128},
//                                   priv<uval,val>{});
//    auto sc_sum = mngr.spawn_new(prog_sc, "sumBlockCounts", ndrange_128_sum,
//                                 in<uval,mref>{},
//                                 out<uval,mref>{wi_one},
//                                 local<uval>{128},
//                                 priv<uval,val>{});
//    auto sc_move = mngr.spawn_new(prog_sc, "moveValidElementsStaged",
//                                  ndrange_128,
//                                  out<uval,val>{one},
//                                  in<uval,val>{},
//                                  out<uval,mref>{k_once},
//                                  in<uval,mref>{},
//                                  in<uval,mref>{},
//                                  local<uval>{128},
//                                  local<uval>{128},
//                                  local<uval>{128},
//                                  priv<uval,val>{});
//    scoped_actor self{system};
//    self->send(sc_count, heads, static_cast<uval>(n));
//    uref blocks;
//    uref valid;
//    self->receive([&](uref& b, uref& v) {
//      blocks = b;
//      valid = v;
//    });
////    cout << "Received blocks" << endl;
////    self->send(sc_sum, blocks, static_cast<uval>(m));
////    self->receive([&](uref& summed) {
////      blocks = summed;
////    });
//    self->send(sc_move, values, valid, blocks, static_cast<uval>(n));
//    self->receive([&](uvec& size, uref& data) {
//      auto vec = data.data(size.front());
//      vec = data.data(size.front());
//      vec = data.data(size.front());
//      cout << "Created compacted array of " << size.front() << " values." << endl;
//      for (auto e : *vec)
//        cout << setw(6) << e << " ";
//      cout << endl;
//    });
//  }
  // clean up
  system.await_all_actors_done();
}

CAF_MAIN()
