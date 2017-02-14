
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

using vec = std::vector<uint32_t>;

using add_atom = atom_constant<atom("add")>;
using init_atom = atom_constant<atom("init")>;
using quit_atom = atom_constant<atom("quit")>;
using index_atom = atom_constant<atom("index")>;

constexpr const char* kernel_file_01 = "./include/sort_rids_by_value.cl";
/*
constexpr const char* kernel_file_02 = "./include/produce_chunck_id_literals.cl";
constexpr const char* kernel_file_03 = "./include/merge_lit_by_val_chids.cl";
constexpr const char* kernel_file_04 = "./include/produce_fills.cl";
constexpr const char* kernel_file_05 = "./include/fuse_fill_literals.cl";
constexpr const char* kernel_file_06 = "./include/compute_colum_length.cl";
*/

/*
constexpr const char* kernel_name_01a = "kernel_wah_index";

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

string decoded_bitmap(const vec& bitmap) {
  if (bitmap.empty()) {
    return "";
  }
  stringstream s;
  for (auto& block : bitmap) {
    if (block & (0x1 << 31)) {
      uint32_t mask = 0x1;
      for (int i = 0; i < 31; ++i) {
        s << ((block & mask) ? '1' : '0');
        mask <<= 1;
      }
    } else {
      auto bit = (block & (0x1 << 30)) ? '1' : '0';
      auto times = (block & (~(0x3 << 30)));
      for (uint32_t i = 0; i < times; ++i) {
        for (uint32_t j = 0; j < 31; ++j) {
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

// For testing, TODO: DELTE THIS
/*
vector<uint32_t> sort_rids_by_value(vector<uint32_t>& input) {
  vector<uint32_t> rids(input.size());
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
  return rids;
}
*/

} // namespace <anonymous>

class config : public actor_system_config {
public:
  string filename = "";
  uint32_t bound = 0;
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

void caf_main(actor_system& system, const config& cfg) {
  vec values;
  if (cfg.filename.empty()) {
    values = {10,  7, 22,  6,  7,  1,  9, 42,  2,  5,
              13,  3,  2,  1,  0,  1, 18, 18,  3, 13,
               5,  9,  0,  3,  2, 19,  5, 23, 22, 10,
               6, 22};
  } else {
    cout << "Reading data from '" << cfg.filename << "' ... " << flush;
    ifstream source{cfg.filename, std::ios::in};
    uint32_t next;
    while (source >> next) {
      values.push_back(next);
    }
  }
  cout << "'" << values.size() << "' values." << endl;
  auto bound = cfg.bound;
  if (bound == 0 && !values.empty()) {
    auto itr = max_element(values.begin(), values.end());
    bound = *itr;
  }
  cout << "Maximum value is '" << bound << "'." << endl;

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
  auto prog_rids   = mngr.create_program_from_file(kernel_file_01, "", dev);
  /*
  auto prog_chunks = mngr.create_program_from_file(kernel_file_02, "", dev);
  auto prog_merge  = mngr.create_program_from_file(kernel_file_03, "", dev);
  auto prog_fills  = mngr.create_program_from_file(kernel_file_04, "", dev);
  auto prog_fuse   = mngr.create_program_from_file(kernel_file_05, "", dev);
  auto prog_colum  = mngr.create_program_from_file(kernel_file_06, "", dev);
  */

  // create spawn configuration
  auto index_space      = spawn_config{dim_vec{values.size()}};
  auto index_space_half = spawn_config{dim_vec{values.size() / 2}};

  // buffers for execution
  vec config{static_cast<uint32_t>(values.size())};
  auto vals_ref = dev.copy_to_device(buffer_type::input_output, values);
  // should be scratch space
  auto rids_ref = dev.copy_to_device(buffer_type::output, vec(),
                                     values.size());
  {
    // create phases
    auto w_rids_1 = mngr.spawn_phase<vec, vec, vec>(prog_rids, "create_rids",
                                                    index_space);
    auto w_rids_2 = mngr.spawn_phase<vec, vec, vec>(prog_rids,
                                                    "ParallelBitonic_B2",
                                                    index_space_half);

    // kernel executions
    scoped_actor self{system};
    auto conf_ref = dev.copy_to_device(buffer_type::input, config);
    self->send(w_rids_1, conf_ref, vals_ref, rids_ref);
    self->receive(
      [&](mem_ref<uint32_t>&, mem_ref<uint32_t>&, mem_ref<uint32_t>&) {
        // nop
      }
    );
    config.resize(2);
    for (uint32_t length = 1; length < values.size(); length <<= 1) {
      int inc = length;
      bool done = false;
      config[0] = inc;
      config[1] = length << 1;
      conf_ref = dev.copy_to_device(buffer_type::input, config);
      self->send(w_rids_2, conf_ref, vals_ref, rids_ref);
      self->receive_while([&] { return !done; })(
        [&](mem_ref<uint32_t>& conf, mem_ref<uint32_t>& vals,
            mem_ref<uint32_t>& rids) {
          inc >>= 1;
          if (inc > 0) {
            config[0] = inc;
            conf = dev.copy_to_device(buffer_type::input, config);
            self->send(w_rids_2, conf, vals, rids);
          } else {
            done = true;
          }
        }
      );
    }

    // test stuff
    /*
    vec test_keys = values;
    vec test_vals = sort_rids_by_value(test_keys);
    */
  }
  // clean up
  system.await_all_actors_done();
}

CAF_MAIN()
