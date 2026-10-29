// Force-included when building ALSRAC (SJTU-ECTL/ALSRAC) with GCC 13: its
// sources rely on standard headers that older libstdc++ pulled in
// transitively.  See README, "ALS baseline".
#include <list>
#include <deque>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <stack>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <utility>
#include <tuple>
#include <bitset>
