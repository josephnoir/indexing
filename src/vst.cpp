
#include <cmath>
#include <random>
#include <vector>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <iostream>
#include <algorithm>

#include "caf/all.hpp"

#include "vast/coder.hpp"
#include "vast/operator.hpp"
#include "vast/wah_bitmap.hpp"
#include "vast/bitmap_index.hpp"

#include "vast/concept/printable/to_string.hpp"
#include "vast/concept/printable/vast/bitmap.hpp"

using namespace std;
using namespace caf;
using namespace vast;

class config : public actor_system_config {
public:
  string filename = "";
  uint32_t bound = 0;
  config() {
    opt_group{custom_options_, "global"}
    .add(filename, "data-file,f", "File to read test data from (one value per line)")
    .add(bound, "bound,b", "maximum value in the set (0 will scan values)");
  }
};

void caf_main(actor_system&, const config& cfg) {
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
  auto amount = values.size();
  cout << "Read " << amount << " values." << endl;
  auto bound = cfg.bound;
  if (bound == 0 && amount > 0) {
    auto itr = max_element(values.begin(), values.end());
    bound = *itr;
  }
  cout << "Maximum value is  " << bound << "." << endl;

  // Extract key set
  vector<uint32_t> keys = values;
  std::sort(keys.begin(), keys.end());
  auto last = std::unique(keys.begin(), keys.end());
  keys.erase(last, end(keys));

  // MEASURE: from
  bitmap_index<uint32_t, equality_coder<wah_bitmap>> bmi{bound + 1};
  for (auto& val : values)
    bmi.push_back(val);
  // MEASURE: to

  // print index by key
  auto& coder = bmi.coder();
  auto& storage = coder.storage();
  for (auto& key : keys)
    cout << "Index for value " << key << ":" << endl
         << to_string(storage[key]) << endl;
}

CAF_MAIN()
