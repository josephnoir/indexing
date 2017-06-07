/******************************************************************************
 * Copyright (C) 2017                                                         *
 * Raphael Hiesgen <raphael.hiesgen (at) haw-hamburg.de>                      *
 *                                                                            *
 * Distributed under the terms and conditions of the BSD 3-Clause License.    *
 *                                                                            *
 * If you did not receive a copy of the license files, see                    *
 * http://opensource.org/licenses/BSD-3-Clause and                            *
 ******************************************************************************/

#include <random>
#include <iostream>

#include "caf/all.hpp"

using namespace std;
using namespace caf;

class config : public actor_system_config {
public:
  size_t from   =         0;
  size_t to     = std::numeric_limits<uint16_t>::max();
  size_t amount = 268500000;
  string separator =   "\n";

  config() {
    opt_group{custom_options_, "global"}
    .add(from, "from,f", "set minimum value (default: 0)")
    .add(to,   "to,t", "set maximum value (default: max_value<uint32>)")
    .add(amount, "amount,a", "set number of values (default: 268 500 000)")
    .add(separator, "separator,s", "separator between numbers (default: \\n)");
  }
};

int main(int argc, char** argv) {
  config cfg;
  // read CLI options
  cfg.parse(argc, argv);
  // return immediately if a help text was printed
  if (cfg.cli_helptext_printed)
    return 0;

  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_int_distribution<uint32_t> dist(cfg.from,cfg.to);
  vector<uint32_t> values(cfg.amount);
  auto& sep = cfg.separator;
  for (size_t i = 0; i < cfg.amount; ++i) {
    cout << dist(rng);
    if (i < cfg.amount - 1) {
      cout << sep;
    }
  }
  cout << endl;
}
