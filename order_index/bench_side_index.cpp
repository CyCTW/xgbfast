// 粗略微基準：比較 SideIndex 與 std::map 在「大部分接近市價、偶爾極端價位」工作負載下的每次操作耗時。
// 僅供相對比較；實際延遲請在目標機器上綁核心量測。
#include "side_index.hpp"

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <map>
#include <memory>
#include <random>
#include <vector>

using namespace oms;

struct Op {
  int64_t px;
  int64_t qty;
  bool add;
};

// 產生穩態工作負載：維持約 targetLevels 個價位，5% 積極價、5% 遠端價
std::vector<Op> makeOps(size_t n, int targetLevels, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::map<int64_t, int> live;  // px -> 訂單數
  std::vector<Op> ops;
  ops.reserve(n);
  const int64_t mid = 1'000'000;
  auto rnd = [&](int64_t lo, int64_t hi) { return std::uniform_int_distribution<int64_t>(lo, hi)(rng); };
  while (ops.size() < n) {
    const bool add = live.empty() || ((int)live.size() < targetLevels ? rnd(0, 99) < 70 : rnd(0, 99) < 40);
    if (add) {
      const int k = (int)rnd(0, 99);
      int64_t px = k < 90 ? mid + rnd(-targetLevels, targetLevels)
                 : k < 95 ? mid + rnd(200, 1'000'000)
                          : mid - rnd(200, 1'000'000);
      ops.push_back({px, 1, true});
      ++live[px];
    } else {
      auto it = live.begin();
      std::advance(it, rnd(0, (int64_t)live.size() - 1));
      ops.push_back({it->first, 1, false});
      if (--it->second == 0) live.erase(it);
    }
  }
  return ops;
}

template <class F>
double timeIt(const std::vector<Op>& ops, int reps, F&& f) {
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    f();
    auto t1 = std::chrono::steady_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / ops.size();
    if (ns < best) best = ns;
  }
  return best;
}

int main() {
  for (int levels : {8, 32, 128}) {
    const auto ops = makeOps(1'000'000, levels, 42);
    volatile int64_t sink = 0;

    const double tIdx = timeIt(ops, 5, [&] {
      auto idx = std::make_unique<SideIndex<Side::Buy, 1024>>();
      for (const Op& o : ops) {
        if (o.add) {
          if (!idx->addOrder(o.px, o.qty)) std::abort();  // 工作負載不應超過容量
        } else {
          idx->reduce(o.px, o.qty, true);
        }
        if (!idx->empty()) sink = sink + idx->bestPx();
      }
    });

    const double tMap = timeIt(ops, 5, [&] {
      std::map<int64_t, std::pair<int64_t, uint32_t>> m;
      for (const Op& o : ops) {
        if (o.add) {
          auto& L = m[o.px];
          L.first += o.qty;
          ++L.second;
        } else {
          auto it = m.find(o.px);
          it->second.first -= o.qty;
          if (--it->second.second == 0) m.erase(it);
        }
        if (!m.empty()) sink = sink + m.rbegin()->first;
      }
    });
    std::printf("~%3d levels: SideIndex %6.1f ns/op   std::map %6.1f ns/op\n", levels, tIdx, tMap);
  }
  return 0;
}
