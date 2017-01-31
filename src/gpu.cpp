
#include <cmath>
#include <random>
#include <vector>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <iostream>
#include <iterator>
#include <algorithm>

#include "caf/all.hpp"
#include "caf/opencl/all.hpp"

using namespace std;
using namespace caf;
using namespace caf::opencl;

namespace {

using init_atom = atom_constant<atom("init")>;
using index_atom = atom_constant<atom("index")>;

constexpr const char* kernel_name = "kernel_wah_index";
constexpr const char* kernel_file = "kernel_wah_bitindex.cl";
// constexpr const char* kernel_source = R"__( /* empty*/  )__";

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

// string decoded_string(vector<uint32_t> bitmap) { }

struct indexer_state {
  actor idx_worker;
  vector<uint32_t> input;
  uint32_t values_left;
  uint32_t in_progress;
  uint32_t concurrently;
  uint32_t batch_size;
};

behavior indexer(stateful_actor<indexer_state>* self) {
  return {
    [=] (init_atom, actor gpu_indexer, vector<uint32_t> input,
         uint32_t batch_size, uint32_t concurrently) {
      self->state.idx_worker = gpu_indexer;
      self->state.input = std::move(input);
      self->state.values_left = input.size();
      self->state.in_progress = 0;
      self->state.concurrently = concurrently;
      self->state.batch_size = batch_size;
      // MEASURE: start
    },
    [=] (index_atom) {
      auto& s = self->state;
      if (s.in_progress >= s.concurrently) {
        cout << "Already " << s.in_progress << " jobs in progress." << endl;
        return;
      }
      auto from = begin(s.input);
      auto remaining = static_cast<uint32_t>(distance(from, end(s.input)));
      if (remaining == 0) {
        cout << "Complete input sent to GPU." << endl;
        return;
      }
      auto advance_by = min(remaining, s.batch_size);
      auto to = from + advance_by;

      vector<uint32_t> input{make_move_iterator(from),
                             make_move_iterator(to)};
      uint32_t num_values = input.size();
      vector<uint32_t> config{
        num_values,      // input
        num_values * 2,  // index
        num_values       // processed values
      };

      // send both matrices to the actor and wait for a result
      self->send(s.idx_worker, std::move(config), move(input));
      s.input.erase(begin(s.input), begin(s.input) + advance_by);
      s.in_progress += 1;
      self->send(self, index_atom::value);
    },
    [=](const vector<uint32_t>& config, const vector<uint32_t>& input,
        const vector<uint32_t>& index,  const vector<uint32_t>& offsets) {
      self->state.in_progress -= 1;
      // MEASURE: to <-- if all values returned

  /*
      cout << "conf: " << config.size() << endl
           << "inpt: " << input.size() << endl
           << "indx: " << index.size() << endl
           << "offs: " << offsets.size() << endl;

      auto length = config[4];
      cout << "index length = " << length << endl;
      for (uint32_t i = 0; i < length; ++i) {
        //cout << as_binary(input[i]) << endl;
        cout << as_binary(input[i]) << " :: "
             << as_binary(index[i]) << " :: "
             << as_binary(offsets[i]) << endl;
      }
  */

      auto keycnt = config[0];
      auto index_length = config[1];
      auto processed = config[2];
      cout << "Created index for " << self->state.batch_size
           << " values, with "     << index_length
           << " blocks and "       << keycnt
           << " keys."             << endl;

      // Searching for some sensible output format
      for (size_t i = 0; i < keycnt; ++i) {
        auto value = input[i];
        auto offset = offsets[i];
        auto length = (i == keycnt - 1) ? index_length - offsets[i]
                                        : offsets[i + 1] - offsets[i];
        cout << "Index for value " << value << " has '" << length
             << "' blocks at offset '" << offset << "':" << endl;
        for (size_t j = 0; j < length; ++j) {
          cout << as_binary(index[offset + j]) << " ";
        }
        cout << endl << endl;
      }
      self->state.values_left -= processed;
      if (self->state.values_left > 0) {
        self->quit();
      } else {
        self->send(self, index_atom::value);
      }
    }
  };
}

} // namespace <anonymous>

class config : public actor_system_config {
public:
  string filename = "";
  uint32_t bound = 0;
  string device_name = "GeForce GT 650M";
  uint32_t batch_size = 1024;
  uint32_t concurrently = 0;
  config() {
    load<opencl::manager>();
    add_message_type<vector<uint32_t>>("data_vector");
    opt_group{custom_options_, "global"}
    .add(filename, "data-file,f", "File with test data (one value per line)")
    .add(bound, "bound,b", "maximum value in the set (0 will scan values)")
    .add(device_name, "device,d", "Device for computation (GeForce GT 650M)")
    .add(batch_size, "batch-size,b", "Values indexed in one batch (1024)")
    .add(concurrently, "concurrently,c", "Concurrent batches sent to GPU "
                                         "(available compute units)");
  }
};

void caf_main(actor_system& system, const config& cfg) {
  vector<uint32_t> values;
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
  cout << "Read '" << values.size() << "' values." << endl;
  auto bound = cfg.bound;
  if (bound == 0 && !values.empty()) {
    auto itr = max_element(values.begin(), values.end());
    bound = *itr;
  }
  cout << "Maximum value is  '" << bound << "'." << endl;

  // read cl kernel file
  auto filename = string("./include/") + kernel_file;
  cout << "Reading source from '" << filename << "' ... " << flush;
  ifstream read_source{filename, std::ios::in};
  string source_contents;
  if (read_source) {
      read_source.seekg(0, std::ios::end);
      source_contents.resize(read_source.tellg());
      read_source.seekg(0, std::ios::beg);
      read_source.read(&source_contents[0], source_contents.size());
      read_source.close();
  } else {
      cout << strerror(errno) << "[!!!] " << endl;
      return;
  }
  cout << "DONE" << endl;

  // create GPU worker
  auto& mngr = system.opencl_manager();
  auto opt = mngr.get_device_if([&](const device& dev){
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
  auto concurrently = cfg.concurrently == 0 ? dev.get_max_compute_units()
                                            : cfg.concurrently;
  auto batch_size = min(cfg.batch_size, static_cast<uint32_t>(values.size()));
  auto double_size = [](const vector<uint32_t>&, const vector<uint32_t>& in) {
    return in.size() * 2;
  };
  auto normal_size = [](const vector<uint32_t>&, const vector<uint32_t>& in) {
    return in.size();
  };
  auto worker = system.opencl_manager().spawn(
    mngr.create_program(source_contents.c_str(), "", dev),
    kernel_name, spawn_config{dim_vec{batch_size}},
    in_out<vector<uint32_t>>{},             // config
    in_out<vector<uint32_t>>{},             // input
    out<vector<uint32_t>>{double_size},     // index
    out<vector<uint32_t>>{normal_size},     // offsets
    buffer<vector<uint32_t>>{normal_size},  // rids
    buffer<vector<uint32_t>>{normal_size},  // chids
    buffer<vector<uint32_t>>{normal_size}   // lits
  );
  auto idx_manger = system.spawn(indexer);
  anon_send(idx_manger, init_atom::value, worker, values, batch_size,
            concurrently);
  anon_send(idx_manger, index_atom::value);
  system.await_all_actors_done();
}

CAF_MAIN()
