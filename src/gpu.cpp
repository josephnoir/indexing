
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

using add_atom = atom_constant<atom("add")>;
using init_atom = atom_constant<atom("init")>;
using quit_atom = atom_constant<atom("quit")>;
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

struct bitmap_index {
  uint32_t bound;
  unordered_map<uint32_t,vector<uint32_t>> index;
  high_resolution_clock::time_point start;
  bool print_results;
};

behavior merger(stateful_actor<bitmap_index>* self) {
  return {
    [=](init_atom, bool print_results, uint32_t bound) {
      self->state.start = high_resolution_clock::now();
      self->state.print_results = print_results;
      self->state.bound = bound;
    },
    [=](add_atom, uint32_t keycnt, uint32_t length,
        vector<uint32_t> keys, vector<uint32_t> offsets,
        vector<uint32_t> index) {
      static_cast<void>(keycnt);
      static_cast<void>(length);
      static_cast<void>(keys);
      static_cast<void>(offsets);
      static_cast<void>(index);
      /*
      cout << "Index has " << length << " elements with "
          << keycnt << " keys" << endl;
      uint32_t next = 0;
      for (size_t i = 0; i < keycnt; ++i) {
        auto key = keys[i];
        auto offset = offsets[i];
        auto len = (i == keycnt - 1) ? length         - offsets[i]
                                     : offsets[i + 1] - offsets[i];
        vector<uint32_t> tmp{index.begin() + offset,
                             index.begin() + offset + len};
        if (key != next) {
          for (uint32_t j = next; j < key; ++j) {
            cout << j << "\t" << endl;
          }
        }
        cout << key << '\t' << decoded_bitmap(tmp) << endl;
        next = key + 1;
      }
      */
      // Switch to VAST bitmap index, built in two stps,
      // first bitmaps than encoder
      for (size_t i = 0; i < keycnt; ++i) {
        auto key = keys[i];
        auto offset = offsets[i];
        auto len = (i == keycnt - 1) ? length - offsets[i]
                                     : offsets[i + 1] - offsets[i];
        auto& dex = self->state.index;
        dex[key].insert(dex[key].end(),
                        make_move_iterator(index.begin() + offset),
                        make_move_iterator(index.begin() + offset + len));
      }
    },
    [=](quit_atom) {
      auto stop = chrono::high_resolution_clock::now();
      if (self->state.print_results) {
        auto dex = self->state.index;
        for (uint32_t key = 0; key <= self->state.bound; ++key) {
          cout << key << '\t' << decoded_bitmap(dex[key]) << endl;
        }
      }
      cout << "Time: '"
           << duration_cast<milliseconds>(stop - self->state.start).count()
           << "' ms" << endl;
      self->quit();
    }
  };
}

struct indexer_state {
  actor idx_worker;
  vector<uint32_t> input;
  uint32_t bound;
  uint32_t input_left;
  uint32_t remaining;
  uint32_t in_progress;
  uint32_t jobs;
  uint32_t wg_num;
  uint32_t wg_size;
};

behavior indexer(stateful_actor<indexer_state>* self, const actor& idx_merger) {
  return {
    [=] (init_atom, actor gpu_indexer, vector<uint32_t> input,
         uint32_t jobs, uint32_t wg_num, uint32_t wg_size,
         bool print_results, uint32_t bound) {
      self->state.idx_worker  = gpu_indexer;
      self->state.remaining   = input.size();
      self->state.input       = std::move(input);
      self->state.in_progress = 0;
      self->state.jobs        = jobs;
      self->state.wg_num      = wg_num;
      self->state.wg_size     = wg_size;
      self->state.bound       = bound;
      self->send(idx_merger, init_atom::value, print_results, bound);
    },
    [=] (index_atom) {
      auto& s = self->state;
      if (s.in_progress >= s.jobs || s.input.empty())
        return;
      auto remaining = static_cast<uint32_t>(s.input.size());
      auto batch_size = min(remaining, s.wg_size * s.wg_num);
      vector<uint32_t> input{make_move_iterator(begin(s.input)),
                             make_move_iterator(begin(s.input) + batch_size)};
      // The config contains the total number of values followed by the number
      // of work groups. For each work group, the config contains the size
      // of the input values and reserved index.
      auto full = batch_size / s.wg_size;
      auto partial = batch_size - (full * s.wg_size);
      vector<uint32_t> config(2 + (full * 2));
      config[0] = batch_size;
      config[1] = full;
      for (uint32_t i = 0; i < full; ++i) {
        config[2 * i + 2] = s.wg_size;
        config[2 * i + 3] = s.wg_size * 2;
      }
      if (partial > 0) {
        config[1] += 1;
        config.emplace_back(partial);
        config.emplace_back(partial * 2);
      }
      //cout << "Config for " << batch_size << " values:" << endl;
      //for (size_t i = 0; i < config.size(); ++i)
      //  cout << "> config[" << i << "]: " << config[i] << endl;
      // send both matrices to the actor and wait for a result
      self->send(s.idx_worker, std::move(config), move(input));
      s.input.erase(begin(s.input), begin(s.input) + batch_size);
      s.in_progress += 1;
      self->send(self, index_atom::value);
    },
    [=](vector<uint32_t> config, vector<uint32_t> keys,
        vector<uint32_t> index,  vector<uint32_t> offsets) {
      auto& s = self->state;
      auto processed = config[0];
      auto wg_num    = config[1];
      s.in_progress -= 1;
      s.remaining   -= processed;
      // split the calculated indecies into independent indexes
      // *maybe*, move this to the merger actor.
      for (uint32_t i = 0; i < wg_num; ++i) {
        auto keycnt = config[2 * i + 2];
        auto length = config[2 * i + 3];
        auto buffer_length = min(processed, s.wg_size);
        auto from = i * s.wg_size;
        auto to = (i * s.wg_size) + buffer_length;
        vector<uint32_t> wg_keys{make_move_iterator(begin(keys) + from),
                                 make_move_iterator(begin(keys) + to)};
        vector<uint32_t> wg_offeset{make_move_iterator(begin(offsets) + from),
                                    make_move_iterator(begin(offsets) + to)};
        vector<uint32_t> wg_index{make_move_iterator(begin(index) + (from * 2)),
                                  make_move_iterator(begin(index) + (to * 2))};
        self->send(idx_merger, add_atom::value, keycnt, length,
                   move(wg_keys), move(wg_offeset), move(wg_index));
        processed -= buffer_length;
      }
      if (s.remaining > 0) {
        self->send(self, index_atom::value);
      } else {
        self->send(idx_merger, quit_atom::value);
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
  string device_name = "GeForce GT 650M";
  uint32_t batch_size = 1023;
  uint32_t jobs = 1;
  uint32_t work_groups = 1;
  bool print_results;
  config() {
    load<opencl::manager>();
    add_message_type<vector<uint32_t>>("data_vector");
    opt_group{custom_options_, "global"}
    .add(filename, "data-file,f", "file with test data (one value per line)")
    .add(bound, "bound,b", "maximum value (0 will scan values)")
    .add(device_name, "device,d", "device for computation (GeForce GTX 780M)")
    .add(batch_size, "batch-size,b", "values indexed in one batch (1023)")
    .add(print_results, "print,p", "print resulting bitmap index")
    .add(work_groups, "work-groups,w", "Use work-groups")
    .add(jobs, "jobs,j", "jobs sent to GPU (1)");
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
  cout << "'" << values.size() << "' values." << endl;
  auto bound = cfg.bound;
  if (bound == 0 && !values.empty()) {
    auto itr = max_element(values.begin(), values.end());
    bound = *itr;
  }
  cout << "Maximum value is '" << bound << "'." << endl;

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
  auto opt = mngr.get_device_if([&](const device_ptr dev) {
      if (cfg.device_name.empty())
        return true;
      return dev->get_name() == cfg.device_name;
  });
  if (!opt) {
    cerr << "Device " << cfg.device_name << " not found." << endl;
    return;
  } else {
    cout << "Using device named '" << (*opt)->get_name() << "'." << endl;
  }
  auto dev = *opt;
  auto jobs = cfg.jobs == 0 ? dev->get_max_compute_units() : cfg.jobs;
  auto batch_size = min(cfg.batch_size, static_cast<uint32_t>(values.size()));
  auto wg_size = min(batch_size,
                     static_cast<uint32_t>(dev->get_max_work_group_size()));
  auto wg_num = cfg.work_groups;
  auto gl_size = wg_size * wg_num; // must be multiple of wg_size
  auto double_size = [](const vector<uint32_t>&, const vector<uint32_t>& in) {
    return in.size() * 2;
  };
  auto normal_size = [](const vector<uint32_t>&, const vector<uint32_t>& in) {
    return in.size();
  };
  //cout << "Creating OpenCL actor with (" << gl_size << ", 1, 1) work items "
  //        "distributed in work groups in batches of (" << wg_size << ", 1, 1)"
  //     << endl;
  auto worker = system.opencl_manager().spawn(
    mngr.create_program(source_contents.c_str(), "", dev),
    kernel_name,
    spawn_config{dim_vec{gl_size}, {}, dim_vec{wg_size}},
    in_out<vector<uint32_t>>{},             // config
    in_out<vector<uint32_t>>{},             // input
    out<vector<uint32_t>>{double_size},     // index
    out<vector<uint32_t>>{normal_size},     // offsets
    scratch<vector<uint32_t>>{normal_size},  // rids
    scratch<vector<uint32_t>>{normal_size},  // chids
    scratch<vector<uint32_t>>{normal_size}   // lits
  );

  auto idx_merger = system.spawn(merger);
  auto idx_manger = system.spawn(indexer, idx_merger);
  anon_send(idx_manger, init_atom::value, worker, values, jobs,
            wg_num, wg_size, cfg.print_results, bound);
  anon_send(idx_manger, index_atom::value);
  system.await_all_actors_done();
}

CAF_MAIN()
