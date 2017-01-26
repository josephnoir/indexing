
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
#include <algorithm>

#include "caf/all.hpp"
#include "caf/opencl/all.hpp"

using namespace std;
using namespace caf;
using namespace caf::opencl;

namespace {

constexpr const char* kernel_name = "kernel_wah_index";
constexpr const char* kernel_file = "kernel_wah_bitindex.cl";
// constexpr const char* kernel_source = R"__( /* empty*/  )__";

template<class T>
string as_binary(T num) {
  stringstream s;
  auto num_bits = (sizeof(T) * 8);
  T mask = T(0x1) << (num_bits - 1);
  while (mask > 0) {
    s << ((num & mask) ? "1" : "0");
    mask >>= 1;
  }
  return s.str();
}

void indexer(event_based_actor* self,
             const string& kernel,
             const string& func) {
  vector<uint32_t> values{10,  7, 22,  6,  7,
                           1,  9, 42,  2,  5,
                          13,  3,  2,  1,  0,
                           1, 18, 18,  3, 13};
  auto amount = values.size();
  /*
  size_t amount = 10;
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_int_distribution<uint32_t> dist(0,10);
  vector<uint32_t> values(amount);
  for (size_t i = 0; i < amount; ++i)
    values[i] = dist(rng);
  */
  for (auto& val : values)
    cout << val << " ";
  cout << endl;

  vector<uint32_t> input = values;
  vector<uint32_t> index(amount * 2);
  vector<uint32_t> offsets(amount);

  vector<uint32_t> config{static_cast<uint32_t>(amount),      // input
                          static_cast<uint32_t>(amount),      // rids
                          static_cast<uint32_t>(amount),      // chids
                          static_cast<uint32_t>(amount),      // lits
                          static_cast<uint32_t>(amount * 2),  // index
                          static_cast<uint32_t>(amount)};     // offsets

  string device_name = "GeForce GT 650M";
  auto& mngr = self->system().opencl_manager();
  auto opt = mngr.get_device_if([device_name](const device& dev){
      return dev.get_name() == device_name;
  });
  if (!opt) {
    cerr << "Device not found." << endl;
    return;
  }
  auto dev = *opt;
  auto double_size = [amount](const vector<uint32_t>&, const vector<uint32_t>&) {
    return amount * 2;
  };
  auto normal_size = [amount](const vector<uint32_t>&, const vector<uint32_t>&) {
    return amount;
  };
  auto worker = self->system().opencl_manager().spawn(
    mngr.create_program(kernel.c_str(), "", dev), func.c_str(),
    spawn_config{dim_vec{amount}},
    in_out<vector<uint32_t>>{},         // config
    in_out<vector<uint32_t>>{},         // input
    out<vector<uint32_t>>{double_size},  // index
    out<vector<uint32_t>>{normal_size}, // offsets
    buffer<vector<uint32_t>>{amount},   // rids
    buffer<vector<uint32_t>>{amount},   // chids
    buffer<vector<uint32_t>>{amount}    // lits
  );
  // send both matrices to the actor and wait for a result
  self->request(worker, chrono::seconds(30), move(config), move(input)).then(
    [amount](const vector<uint32_t>& config, const vector<uint32_t>& input,
             const vector<uint32_t>& index, const vector<uint32_t>& offsets) {
      static_cast<void>(input);
      static_cast<void>(index);
      static_cast<void>(offsets);
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


      auto index_length = config[4];
      auto keycnt = config[0];
      cout << "Created index for " << amount
           << " values, with " << index_length
           << " blocks and " << keycnt
           << " keys." << endl;

/*
      // Searching for some sensible output format
      for (size_t i = 0; i < keycnt; ++i) {
        auto value = input[i];
        auto offset = offsets[i];
        auto length = (i == keycnt - 1) ? index_length - offsets[i]
                                          : offsets[i + 1] - offsets[i];
        cout << "Index for value " << value << ":" << endl
             << "> length " << length << endl << "> offset " << offset << endl;
        for (size_t j = 0; j < length; ++j) {
          cout << as_binary(index[offset + j]) << " ";
        }
        cout << endl << endl;
      }
*/
    }
  );
}

} // namespace <anonymous>

int main(void) {
  // TODO introduce cmd line arguments for kernel and function
  actor_system_config cfg;
  cfg.load<opencl::manager>()
     .add_message_type<vector<uint32_t>>("data_vector");
  actor_system system{cfg};
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
      return errno;
  }
  cout << "DONE" << endl;
  system.spawn(indexer, source_contents, kernel_name);
  system.await_all_actors_done();
}

