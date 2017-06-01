
#include <cmath>
#include <tuple>
#include <chrono>
#include <limits>
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

template <class T, class E = typename enable_if<is_integral<T>::value>::type>
T round_up(T numToRound, T multiple)  {
  assert(multiple > 0);
  return ((numToRound + multiple - 1) / multiple) * multiple;
}

template <class T, typename std::enable_if<is_integral<T>{}, int>::type = 0>
uval as_uval(T val) { return static_cast<uval>(val); }


template <class T>
vector<T> compact(vector<T> values, vector<T> heads) {
  assert(values.size() == heads.size());
  vector<T> result;
  auto res_size = count(begin(heads), end(heads), 1u);
  result.reserve(res_size);
  for (size_t i = 0; i < values.size(); ++i) {
    if (heads[i] == 1)
      result.emplace_back(values[i]);
  }
  return result;
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
  string device_name = "GeForce GTX 780M";
  bool print_results;
  double frequency = 0.01;
  config() {
    load<opencl::manager>();
    opt_group{custom_options_, "global"}
    .add(filename, "data-file,f", "file with test data (one value per line)")
    .add(bound, "bound,b", "maximum value (0 will scan values)")
    .add(device_name, "device,d", "device for computation (GeForce GTX 780M, "
                      "empty string will take first available device)")
    .add(print_results, "print,p", "print resulting bitmap index")
    .add(threshold, "threshold,t", "Threshold for output (1500)")
    .add(iterations, "iterations,i", "Number of times the empty kernel is supposed to run (1000)")
    .add(frequency, "frequency,F", "Frequency of 1 in the heads array.(0.1)");
  }
};

} // namespace <anonymous>

/*****************************************************************************\
                                    MAIN!
\*****************************************************************************/

void caf_main(actor_system& system, const config& cfg) {
  uvec values;
  uvec heads;
  random_device rd;  //Will be used to obtain a seed for the random number engine
  mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  uniform_int_distribution<size_t> s_gen(1, 100);
  uniform_int_distribution<size_t> v_gen(1, numeric_limits<uint16_t>::max());
  bernoulli_distribution h_gen(cfg.frequency);
  // ---- get data ----
  if (cfg.filename.empty()) {
    auto size = s_gen(gen) * 1048576 + s_gen(gen);
    cout << "Compacting " << size << " values." << endl;
    values.reserve(size);
    for (size_t i = 0; i < size; ++i)
      values.emplace_back(v_gen(gen));
  } else {
    ifstream source{cfg.filename, std::ios::in};
    uval next;
    while (source >> next) {
      values.push_back(next);
    }
  }
  heads.reserve(values.size());
  heads.emplace_back(1);
  for (size_t i = 1; i < values.size(); ++i)
    heads.emplace_back(h_gen(gen) ? 1 : 0);
  // ---- get device ----
  auto& mngr = system.opencl_manager();
  auto opt = mngr.get_device_if([&](const device_ptr dev) {
      if (cfg.device_name.empty())
        return true;
      return dev->get_name() == cfg.device_name;
  });
  if (!opt) {
    opt = mngr.get_device_if([&](const device_ptr) { return true; });
    if (!opt) {
      cout << "No device found." << endl;
      return;
    }
    cerr << "Using device '" << (*opt)->get_name() << "'." << endl;
  }

  // ---- general ----
  auto dev = move(*opt);
  auto prog_es = mngr.create_program_from_file("./include/scan.cl", "", dev);
  auto prog_sc = mngr.create_program_from_file("./include/stream_compaction.cl",
                                               "", dev);
  {
    // ---- funcs ----
    auto half_block = dev->get_max_work_group_size() / 2;
    auto get_size = [half_block](size_t n) -> size_t {
      return round_up((n + 1) / 2, half_block);
    };
    auto half_size_for = [](size_t n, size_t block) -> size_t {
      return round_up((n + 1) / 2, block);
    };
    auto reduced_scan = [&](const uref&, uval n) {
      // calculate number of groups from the group size from the values size
      return size_t{get_size(n) / half_block};
    };
    auto ndr_scan = [half_size_for, half_block](size_t dim) {
      return spawn_config{dim_vec{half_size_for(dim,half_block)}, {},
                                  dim_vec{half_block}};
    };
    auto ndr_compact = [](uval dim) {
      return spawn_config{dim_vec{round_up(dim, 128u)}, {}, dim_vec{128}};
    };
    auto reduced_compact = [](const uref&, uval n) {
      return size_t{round_up(n, 128u) / 128u};
    };
    auto one = [](uref&, uref&, uref&, uval) { return size_t{1}; };
    auto k_compact = [](uref&, uref&, uref&, uval k) { return size_t{k}; };
    // spawn arguments
    auto ndr = spawn_config{dim_vec{half_block}, {}, dim_vec{half_block}};
    // actors
    // exclusive scan
    auto scan1 = mngr.spawn_new(
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
    auto scan2 = mngr.spawn_new(
      prog_es, "es_phase_2",
      spawn_config{dim_vec{half_block}, {}, dim_vec{half_block}},
      in_out<uval,mref,mref>{},
      in_out<uval,mref,mref>{},
      priv<uval, val>{}
    );
    auto scan3 = mngr.spawn_new(
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
    // stream compaction
    auto sc_count = mngr.spawn_new(
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
    auto sc_move = mngr.spawn_new(
      prog_sc, "moveValidElementsStaged", ndr,
      [ndr_compact](spawn_config& conf, message& msg) -> optional<message> {
        msg.apply([&](const uref&, const uref&, const uref&, uval n) {
          conf = ndr_compact(n);
        });
        return std::move(msg);
      },
      out<uval,mref>{one},
      in_out<uval,mref,mref>{},
      out<uval,mref>{k_compact},
      in_out<uval,mref,mref>{},
      in_out<uval,mref,mref>{},
      local<uval>{128},
      local<uval>{128},
      local<uval>{128},
      priv<uval,val>{}
    );

    auto values_r = dev->global_argument(values);
    auto heads_r = dev->global_argument(heads);
    auto expected = compact(values, heads);
    uref data_r;

    // computations
    scoped_actor self{system};
    self->send(sc_count, heads_r, as_uval(heads_r.size()));
    self->receive([&](uref& blocks, uref& heads) {
      self->send(scan1, blocks, as_uval(blocks.size()));
      heads_r = heads;
    });
    self->receive([&](uref& data, uref& incs) {
      self->send(scan2, data, incs, as_uval(incs.size()));
    });
    self->receive([&](uref& data, uref& incs) {
      self->send(scan3, data, incs, as_uval(data.size()));
    });
    self->receive([&](uref& results) {
      self->send(sc_move, values_r, heads_r, results, as_uval(values_r.size()));
    });
    self->receive([&](uref& size, uref&, uref& data, uref&, uref&) {
      auto size_exp = size.data();
      auto num = (*size_exp)[0];
      auto exp = data.data(num);
      auto actual = std::move(*exp);
      cout << (expected != actual ? "FAILURE" : "SUCCESS") << endl;
    });
  }
  // clean up
  system.await_all_actors_done();
}

CAF_MAIN()
