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

#include "vast/concept/printable/stream.hpp"
#include "vast/concept/printable/to_string.hpp"
#include "vast/concept/printable/vast/coder.hpp"
#include "vast/concept/printable/vast/bitmap.hpp"

using namespace std;
using namespace std::chrono;
using namespace caf;
using namespace vast;

class config : public actor_system_config {
public:
  string filename = "";
  uint32_t bound = 0;
  bool print_results;
  config() {
    opt_group{custom_options_, "global"}
    .add(filename, "data-file,f", "File with test data (one value per line)")
    .add(bound, "bound,b", "maximum value (0 will scan values)")
    .add(print_results, "print,p", "print resulting bitmap index");
  }
};

void caf_main(actor_system&, const config& cfg) {
  std::vector<uint32_t> values;
  if (cfg.filename.empty()) {
    std::cout << "Pleas sepcify a file with inout data using '-f'" << std::endl;
    return;
  } else {
    ifstream source{cfg.filename, std::ios::in};
    uint32_t next;
    while (source >> next)
      values.push_back(next);
  }
  auto amount = values.size();
  auto bound = cfg.bound;
  if (bound == 0 && amount > 0) {
    auto itr = max_element(values.begin(), values.end());
    bound = *itr;
  }

  auto start = high_resolution_clock::now();
  bitmap_index<uint32_t, equality_coder<wah_bitmap>> bmi{bound + 1};
  for (auto& val : values)
    bmi.push_back(val);
  auto stop = high_resolution_clock::now();

  cout << duration_cast<microseconds>(stop - start).count() << " us" << endl;
}

CAF_MAIN()
