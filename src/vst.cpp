
#include <cmath>
#include <random>
#include <vector>
#include <cassert>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <iostream>
#include <algorithm>

#include "vast/bitmap_index.hpp"

using namespace std;
using namespace vast;

namespace {

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

} // namespace <anonymous>

int main(void) {

}

