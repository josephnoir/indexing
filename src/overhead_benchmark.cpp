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
#include "caf/opencl/mem_ref.hpp"

using namespace std;
using namespace std::chrono;
using namespace caf;
using namespace caf::opencl;

namespace {

using uval = uint32_t;
using uvec = std::vector<uval>;
using uref = mem_ref<uval>;

constexpr const char* kernel_file = "./include/empty_kernel.cl";
constexpr const char* kernel_name = "empty_kernel";

class config : public actor_system_config {
public:
  bool use_mapping_functions = false;
  size_t iterations = 1000;
  string device_name = "GeForce GTX 780M";
  config() {
    load<opencl::manager>();
    opt_group{custom_options_, "global"}
    .add(device_name, "device,d", "device for computation (GeForce GTX 780M, "
                      "empty string will take first available device)")
    .add(use_mapping_functions, "use-mapping-functions,m", "Use the mapping"
                                " functions for measurements instead of a message"
                                " round trip including the OpenCL API")
    .add(iterations, "iterations,i", "Number of measurements");
  }
};

} // namespace <anonymous>

void caf_main(actor_system& system, const config& cfg) {
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
    cerr << "No device found." << endl;
    return;
  }
  auto dev = *opt;

  // load kernels
  auto prog_overhead = mngr.create_program_from_file(kernel_file, "", dev);

  // create spawn configuration
  size_t n = 1;
  auto ndr_oh = nd_range{dim_vec{n}};
  // buffers for execution
  uvec data(n);
  std::iota(begin(data), end(data), 0);
  vector<size_t> measurements;
  measurements.reserve(cfg.iterations);
  if (cfg.use_mapping_functions) {
    auto start = high_resolution_clock::now();
    auto stop = start;
    auto stage01 = mngr.spawn(
      prog_overhead, kernel_name, ndr_oh,
      [&](nd_range& , message& msg) -> optional<message> {
        return std::move(msg);
      },
      [&] (uref& result) -> message {
        start = high_resolution_clock::now();
        return make_message(move(result));
      },
      in_out<uval,val,mref>{}
    );
    auto stage02 = mngr.spawn(
      prog_overhead, kernel_name, ndr_oh,
      [&](nd_range&, message& msg) -> optional<message> {
        stop = high_resolution_clock::now();
        measurements.push_back(duration_cast<microseconds>(stop - start).count());
        return std::move(msg);
      },
      in_out<uval,mref,val>{}
    );
    auto pipeline = stage02 * stage01;
    scoped_actor self{system};
    for(size_t i = 0; i < cfg.iterations; ++i) {
      start = high_resolution_clock::now();
      self->send(pipeline, data);
      self->receive([&](uvec&) {
        dev->synchronize();
      });
    }
  } else {
    auto stage = mngr.spawn(prog_overhead, kernel_name, ndr_oh,
                            in_out<uval,mref,mref>{});
    scoped_actor self{system};
    auto start = high_resolution_clock::now();
    auto stop = start;
    auto data_ref = dev->global_argument(data);
    dev->synchronize();
    vector<size_t> measurements(cfg.iterations);
    for(size_t i = 0; i < cfg.iterations; ++i) {
      start = high_resolution_clock::now();
      self->send(stage, data_ref);
      self->receive([&](uref&) {
        stop = high_resolution_clock::now();
      });
      dev->synchronize();
      measurements[i] = duration_cast<microseconds>(stop - start).count();
    }
  }
  size_t mean = accumulate(begin(measurements), end(measurements), 0) / measurements.size();
  cout << "Took " << mean << "us on average over " << measurements.size()
       << " runs." << endl;
  system.await_all_actors_done();
}

CAF_MAIN()
