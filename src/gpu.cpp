
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

string decoded_bitmap(const vector<uint32_t>& bitmap) {
  //cout << endl;
  //for (auto t : bitmap) {
  //  cout << as_binary(t) << " ";
  //}
  //cout << endl;
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
  res.erase(res.find_last_not_of("0") + 1);
  return res;
}

struct indexer_state {
  actor idx_worker;
  vector<uint32_t> input;
  uint32_t bound;
  uint32_t remaining;
  uint32_t in_progress;
  uint32_t concurrently;
  uint32_t batch_size;
  bool print_results;
  unordered_map<uint32_t,vector<uint32_t>> index;
  high_resolution_clock::time_point start;
};

behavior indexer(stateful_actor<indexer_state>* self) {
  return {
    [=] (init_atom, actor gpu_indexer, vector<uint32_t> input,
         uint32_t batch_size, uint32_t concurrently,
         bool print_results, uint32_t bound) {
      self->state.idx_worker = gpu_indexer;
      self->state.remaining = input.size();
      self->state.input = std::move(input);
      self->state.in_progress = 0;
      self->state.concurrently = concurrently;
      self->state.batch_size = batch_size;
      self->state.print_results = print_results;
      self->state.bound = bound;
      self->state.start = high_resolution_clock::now();
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
        num_values * 3,  // index
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
      cout << "Index has " << index_length << " elements with "
           << keycnt << " keys" << endl;
      for (size_t i = 0; i < keycnt; ++i) {
        auto value = input[i];
        auto offset = offsets[i];
        auto length = (i == keycnt - 1) ? index_length - offsets[i]
                                        : offsets[i + 1] - offsets[i];
        cout << "Accessing " << value << " from " << offset
             << " to " << (offset + length) << " (" << length << " blocks) key "
             << i << " of " << keycnt << "." << endl;
        auto& dex = self->state.index;
        dex[value].insert(dex[value].end(),
                          // tmp.begin(), tmp.end());
                          index.begin() + offset,
                          index.begin() + offset + length);
                          //make_move_iterator(index.begin() + offset),
                          //make_move_iterator(index.begin() + offset + length));
      }

      auto processed = config[2];
      self->state.remaining -= processed;
      if (self->state.remaining > 0) {
        self->send(self, index_atom::value);
        cout << "Values left: " << self->state.remaining << endl;
      } else {
        auto stop = chrono::high_resolution_clock::now();
        if (self->state.print_results) {
          auto dex = self->state.index;
          for (uint32_t key = 0; key <= self->state.bound; ++key) {
            //cout << ignore(key) << '\t' << decoded_bitmap(dex[key]) << endl;
            cout << key << '\t' << decoded_bitmap(dex[key]) << endl;
          }
        }
        cout << "Time: '"
             << duration_cast<milliseconds>(stop - self->state.start).count()
             << "' ms" << endl;
        self->quit();
      }
    }
  };
}

} // namespace <anonymous>

class config : public actor_system_config {
public:
  string filename = "";
  uint32_t bound = 0;
  string device_name = "GeForce GTX 780M";
  uint32_t batch_size = 1024;
  uint32_t concurrently = 0;
  bool print_results;
  config() {
    load<opencl::manager>();
    add_message_type<vector<uint32_t>>("data_vector");
    opt_group{custom_options_, "global"}
    .add(filename, "data-file,f", "file with test data (one value per line)")
    .add(bound, "bound,b", "maximum value (0 will scan values)")
    .add(device_name, "device,d", "device for computation (GeForce GTX 780M)")
    .add(batch_size, "batch-size,b", "values indexed in one batch (1024)")
    .add(print_results, "print,p", "print resulting bitmap index")
    .add(concurrently, "concurrently,c", "concurrent batches sent to GPU "
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
    return in.size() * 3;
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

  //unordered_map<uint32_t,vector<uint32_t>> index;
  auto idx_manger = system.spawn(indexer);
  anon_send(idx_manger, init_atom::value, worker, values, batch_size,
            concurrently, cfg.print_results, bound);
  anon_send(idx_manger, index_atom::value);
  system.await_all_actors_done();
}

CAF_MAIN()
