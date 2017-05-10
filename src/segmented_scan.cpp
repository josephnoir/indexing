
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
  random_device rd;  //Will be used to obtain a seed for the random number engine
  mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  uniform_int_distribution<size_t> size_gen(1, 100); //1048576);
  bernoulli_distribution flag_gen(0.1);
  // ---- get data ----
  if (cfg.filename.empty()) {
    values.resize(size_gen(gen) * 1048576 + size_gen(gen));
    iota(values.begin(), values.end(), 1);
    // cout << "A filename with test data is required, see --help for detailts."
    //      << endl;
    // return;
    cout << "Using " << values.size() << " values." << endl;
  } else {
    ifstream source{cfg.filename, std::ios::in};
    uval next;
    while (source >> next) {
      values.push_back(next);
    }
  }
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
  auto prog = mngr.create_program_from_file("./include/segmented_scan.cl",
                                              "", dev);
  // --- scope to ensure actor cleanup ---
  {
    // ---- input parameters ----
    size_t n = values.size();
    size_t group_size = 512;
    // size_t global_range = round_up((n + 1) / 2, group_size);
    // size_t local_range = group_size;
    // size_t groups = (global_range / local_range);
    heads.reserve(values.size());
    heads.emplace_back(1); // should start with a partition
    for (size_t i = 1; i < values.size(); ++i)
      heads.emplace_back(flag_gen(gen) ? 1 : 0);

    // ---- functions for arguments ----
    auto get_size = [group_size](size_t n) -> size_t {
      return round_up((n + 1) / 2, group_size);
    };
    auto nd_conf = [group_size, get_size](size_t dim) {
      return spawn_config{dim_vec{get_size(dim)}, {}, dim_vec{group_size}};
    };
    auto reduced_vec = [&](const uvec&, const uvec&, const uvec&, uval n) {
      // calculate number of groups from the group size from the values size
      return size_t{get_size(n) / group_size};
    };
    auto reduced_ref = [&](const uref&, const uref&, const uref&, uval n) {
      // calculate number of groups from the group size from the values size
      return size_t{get_size(n) / group_size};
    };
    // config for multi-level segmented scan
    uval n_outer = n;
    uval n_inner = get_size(n_outer) / group_size;
    uval n_block = get_size(n_inner) / group_size;
    // ---- ndranges ----
    auto ndr_upsweep_01 = nd_conf(n_outer);
    auto ndr_upsweep_02 = nd_conf(n_inner);
    auto ndr_block = spawn_config{dim_vec{group_size}, {}, dim_vec{group_size}};
    auto ndr_downsweep_02 = ndr_upsweep_02;
    auto ndr_downsweep_01 = ndr_upsweep_01;
    // ---- actors ----
    auto phase1 = mngr.spawn_new(prog, "upsweep", ndr_upsweep_01,
                                 in_out<uval,mref,mref>{},      // data
                                 in_out<uval,val,mref>{},      // partition
                                 in_out<uval,val,mref>{},      // tree
                                 out<uval,mref>{reduced_vec},  // last_data
                                 out<uval,mref>{reduced_vec},  // last_part
                                 out<uval,mref>{reduced_vec},  // last_tree
                                 local<uval>{group_size * 2},  // data buffer
                                 local<uval>{group_size * 2},  // heads buffer
                                 priv<uval, val>{});
    auto phase2 = mngr.spawn_new(prog, "upsweep", ndr_upsweep_02,
                                 in_out<uval,mref,mref>{},     // data
                                 in_out<uval,mref,mref>{},     // partition
                                 in_out<uval,mref,mref>{},     // tree
                                 out<uval,mref>{reduced_ref},  // last_data
                                 out<uval,mref>{reduced_ref},  // last_part
                                 out<uval,mref>{reduced_ref},  // last_tree
                                 local<uval>{group_size * 2},  // data buffer
                                 local<uval>{group_size * 2},  // heads buffer
                                 priv<uval, val>{});
    auto phase3 = mngr.spawn_new(prog, "block_scan", ndr_block,
                                 in_out<uval,mref,mref>{},     // data
                                 in_out<uval,mref,mref>{},     // partition
                                 in<uval,mref>{},              // tree
                                 priv<uval, val>{});           // length
    auto phase4 = mngr.spawn_new(prog, "downsweep", ndr_downsweep_02,
                                 in_out<uval,mref,mref>{},     // data
                                 in_out<uval,mref,mref>{},     // partition
                                 in<uval,mref>{},              // tree
                                 in<uval,mref>{},              // last_data
                                 in<uval,mref>{},              // last_partition
                                 local<uval>{group_size * 2},  // data buffer
                                 local<uval>{group_size * 2},  // part buffer
                                 local<uval>{group_size * 2},  // tree buffer
                                 priv<uval, val>{});
    auto phase5 = mngr.spawn_new(prog, "downsweep_inc", ndr_downsweep_01,
                                 in_out<uval,mref,mref>{},     // data
                                 in_out<uval,mref,mref>{},     // partition
                                 in<uval,mref>{},              // tree
                                 in<uval,mref>{},              // last_data
                                 in<uval,mref>{},              // last_partition
                                 in<uval,mref>{},              // original data
                                 local<uval>{group_size * 2},  // data buffer
                                 local<uval>{group_size * 2},  // part buffer
                                 local<uval>{group_size * 2},  // tree buffer
                                 priv<uval, val>{});
    // ---- test data ----
    //auto scanned = segmented_exclusive_scan(values, heads);
    auto scanned = segmented_inclusive_scan(values, heads);

    // ---- computations -----
    // TODO: use request to avoid the write back to variables
    scoped_actor self{system};
    uref vals = dev->global_argument(values);
    auto save = dev->copy(vals);
    uref d, p, t, d2, p2, t2;
    self->send(phase1, vals, heads, heads, static_cast<uval>(n_outer));
    self->receive([&](uref&      data, uref&      part, uref&      tree,
                      uref& last_data, uref& last_part, uref& last_tree) {
      d = data;
      p = part;
      t = tree;
      self->send(phase2, last_data, last_part, last_tree,
                 static_cast<uval>(n_inner));
    });
    self->receive([&](uref&      data, uref&      part, uref&      tree,
                      uref& last_data, uref& last_part, uref& last_tree) {
      d2 = data;
      p2 = part;
      t2 = tree;
      self->send(phase3, last_data, last_part, last_tree,
                 static_cast<uval>(n_block));
    });
    self->receive([&](uref& ld, uref& lp) {
      self->send(phase4, d2, p2, t2, ld, lp, static_cast<uval>(n_inner));
    });
    self->receive([&](uref& ld, uref& lp) {
      self->send(phase5, d, p, t, ld, lp, *save, static_cast<uval>(n_outer));
    });
    self->receive([&](const uvec& results, uref&) {
      if (results != scanned) {
        /*
        cout << "Expected different result" << endl;
        cout << "   idx   ||    val   | expected | received |" << endl;
        for (size_t i = 0; i < results.size(); ++i) {
          if (heads[i] == 1)
            cout << "---------||----------|----------|----------|------" << endl;
          cout << setw(8) << i          << " || "
               << setw(8) << values[i]  << " | "
               << setw(8) << scanned[i] << " | "
               << setw(8) << results[i] << " | ";
          if (scanned[i] != results[i]) {
            cout << "!!!!";
          }
            cout << endl;
        }
        */
        cout << "Failure" << endl;
      } else {
        cout << "Success" << endl;
      }
    });
    /*
    // This just tested a block-level segmented scan, i.e., phase 2
    self->send(block, values, heads, heads, static_cast<uval>(n));
    self->receive([&](uvec& results, uvec&, uvec&) {
      if (results != scanned) {
        cout << "Expected different result" << endl;
        cout << "idx || val | expected | received |" << endl;
        for (size_t i = 0; i < results.size(); ++i) {
          if (heads[i] == 1)
            cout << "----||-----|----------|----------|------" << endl;
          cout << setw(3) << i          << " || "
               << setw(3) << values[i]  << " | "
               << setw(8) << scanned[i] << " | "
               << setw(8) << results[i] << " | ";
          if (scanned[i] != results[i]) {
            cout << "!!!!";
          }
            cout << endl;
        }
      } else {
        cout << "Success" << endl;
      }
    });
    */
  }
  //dev->queue_count();

  // ---- DONE ----
  system.await_all_actors_done();
}

CAF_MAIN()
