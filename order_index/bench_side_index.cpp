// 價位索引實作比較：
//   de   = SideIndex        （雙端排序陣列，目前採用）
//   back = BackSortedIndex  （單端排序陣列，原方案一）
//   map  = std::map          （對照組）
//
// 每個工作負載都是「新增一張單 / 移除一張單」的序列，維持大約 L 個價位的穩態。
// 量測：
//   * mean  ：整批執行的平均每次操作耗時（不含計時器開銷），取 5 次最佳
//   * p50/p99/p99.9/p99.99：逐筆計時的延遲分布，已扣除計時器本身的開銷；3 次取各百分位最小值
//     （不列 max：在共用 VM 上主要反映中斷與排程，不是資料結構本身）
//
// 僅供相對比較；實際延遲請在目標機器上綁核心量測。
// 用法：bench_side_index [ops]
#include "back_sorted_index.hpp"
#include "side_index.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif
#if defined(__x86_64__)
#include <x86intrin.h>
#endif

using namespace oms;

namespace {

constexpr uint32_t kCap = 1024;  // 兩種陣列都用同樣容量，確保不會因容量不足而失敗

struct Op {
  int64_t px;
  bool add;
};

enum class Workload { Near, Mixed, DeepChurn, AggrChurn, Requeue };

const char* workloadName(Workload w) {
  switch (w) {
    case Workload::Near: return "near";
    case Workload::Mixed: return "mixed";
    case Workload::DeepChurn: return "deep-churn";
    case Workload::AggrChurn: return "aggr-churn";
    case Workload::Requeue: return "requeue";
  }
  return "?";
}

const char* workloadDesc(Workload w) {
  switch (w) {
    case Workload::Near: return "100% 價位在 mid±L，隨機移除";
    case Workload::Mixed: return "90% 近市價、5% 積極、5% 遠端；極端價位傾向很快移除";
    case Workload::DeepChurn: return "70% 近市價、30% 遠端被動（最差端）且很快移除";
    case Workload::AggrChurn: return "70% 近市價、30% 積極價（最佳端）且很快移除";
    case Workload::Requeue: return "最佳價成交後在更差一檔補單（做市：吃掉最佳、最差端補回）";
  }
  return "?";
}

// 以買方為準產生操作序列（積極 = 高於 mid，遠端 = 低於 mid）
std::vector<Op> makeOps(Workload w, size_t n, int L, uint64_t seed) {
  std::mt19937_64 rng(seed);
  auto rnd = [&](int64_t lo, int64_t hi) { return std::uniform_int_distribution<int64_t>(lo, hi)(rng); };
  std::map<int64_t, int> live;  // px → 訂單數
  std::vector<int64_t> extremes;
  std::vector<Op> ops;
  ops.reserve(n);
  const int64_t mid = 1'000'000;

  auto addAt = [&](int64_t px) { ops.push_back({px, true}); ++live[px]; };
  auto removeAt = [&](int64_t px) {
    ops.push_back({px, false});
    auto it = live.find(px);
    if (--it->second == 0) live.erase(it);
  };

  if (w == Workload::Requeue) {
    // 從 [mid-L+1, mid] 每檔一張開始；每次：最佳價成交移除，在目前最差價下一檔補一張。
    // 價位帶會一路往下移（新價位全部落在最差端）。
    for (int i = 0; i < L; ++i) addAt(mid - i);
    while (ops.size() < n) {
      removeAt(live.rbegin()->first);
      addAt(live.begin()->first - 1);
    }
    ops.resize(n);
    return ops;
  }

  while (ops.size() < n) {
    const bool wantAdd = live.empty() || ((int)live.size() < L ? rnd(0, 99) < 70 : rnd(0, 99) < 40);
    if (wantAdd) {
      const int k = (int)rnd(0, 99);
      int64_t px;
      bool extreme = false;
      switch (w) {
        case Workload::Near:
          px = mid + rnd(-L, L);
          break;
        case Workload::Mixed:
          extreme = k >= 90;
          px = k < 90 ? mid + rnd(-L, L) : k < 95 ? mid + rnd(200, 1'000'000) : mid - rnd(200, 1'000'000);
          break;
        case Workload::DeepChurn:
          extreme = k >= 70;
          px = k < 70 ? mid + rnd(-L, L) : mid - rnd(200, 1'000'000);
          break;
        case Workload::AggrChurn:
          extreme = k >= 70;
          px = k < 70 ? mid + rnd(-L, L) : mid + rnd(200, 1'000'000);
          break;
        default:
          px = mid;
      }
      addAt(px);
      if (extreme) extremes.push_back(px);
    } else {
      int64_t px = 0;
      bool picked = false;
      while (!extremes.empty() && rnd(0, 99) < 70) {  // 極端價位很快被刪或成交
        px = extremes.back();
        extremes.pop_back();
        if (live.count(px)) { picked = true; break; }
      }
      if (!picked) {
        auto it = live.begin();
        std::advance(it, rnd(0, (int64_t)live.size() - 1));
        px = it->first;
      }
      removeAt(px);
    }
  }
  return ops;
}

// std::map 轉接成相同 API
struct MapIndex {
  std::map<int64_t, std::pair<int64_t, uint32_t>> m;
  uint32_t addOrder(int64_t px, int64_t qty) {
    auto& L = m[px];
    L.first += qty;
    ++L.second;
    return 0;
  }
  bool reduce(int64_t px, int64_t qty, bool) {
    auto it = m.find(px);
    it->second.first -= qty;
    if (--it->second.second == 0) m.erase(it);
    return true;
  }
  bool empty() const { return m.empty(); }
  int64_t bestPx() const { return m.rbegin()->first; }
};

template <class Index>
inline void apply(Index& idx, const Op& o, int64_t& sink) {
  if (o.add) {
    if (idx.addOrder(o.px, 1) == kInvalid) std::abort();  // 容量足夠，不應發生
  } else {
    idx.reduce(o.px, 1, true);
  }
  if (!idx.empty()) sink += idx.bestPx();  // 模擬策略每次查最佳價
}

inline uint64_t nowTicks() {
#if defined(__x86_64__)
  unsigned aux;
  return __rdtscp(&aux);
#else
  return (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
#endif
}

double ticksPerNs() {
#if defined(__x86_64__)
  const auto t0 = std::chrono::steady_clock::now();
  const uint64_t c0 = nowTicks();
  while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(200)) {}
  const uint64_t c1 = nowTicks();
  const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
  return double(c1 - c0) / ns;
#else
  return 1.0;  // steady_clock 已經是 ns
#endif
}

struct Result {
  double mean, p50, p99, p999, p9999;
};

template <class Index>
Result run(const std::vector<Op>& ops, double tpn, uint64_t timerOverhead) {
  Result r{};
  volatile int64_t out = 0;

  // 平均：整批計時
  double best = 1e30;
  for (int rep = 0; rep < 5; ++rep) {
    auto idx = std::make_unique<Index>();
    int64_t sink = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (const Op& o : ops) apply(*idx, o, sink);
    const auto t1 = std::chrono::steady_clock::now();
    out = out + sink;
    best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count() / ops.size());
  }
  r.mean = best;

  // 分布：逐筆計時。跑 3 次，每個百分位取最小值（VM 干擾只會讓數字變大，取最小值可濾掉）
  r.p50 = r.p99 = r.p999 = r.p9999 = 1e30;
  std::vector<uint64_t> lat(ops.size());
  for (int rep = 0; rep < 3; ++rep) {
    auto idx = std::make_unique<Index>();
    int64_t sink = 0;
    for (size_t i = 0; i < ops.size(); ++i) {
      const uint64_t a = nowTicks();
      apply(*idx, ops[i], sink);
      const uint64_t b = nowTicks();
      const uint64_t d = b - a;
      lat[i] = d > timerOverhead ? d - timerOverhead : 0;
    }
    out = out + sink;
    std::sort(lat.begin(), lat.end());
    auto pct = [&](double p) { return lat[std::min(lat.size() - 1, (size_t)(p * lat.size()))] / tpn; };
    r.p50 = std::min(r.p50, pct(0.50));
    r.p99 = std::min(r.p99, pct(0.99));
    r.p999 = std::min(r.p999, pct(0.999));
    r.p9999 = std::min(r.p9999, pct(0.9999));
  }
  return r;
}

uint64_t measureTimerOverhead() {
  std::vector<uint64_t> v(200000);
  for (auto& x : v) {
    const uint64_t a = nowTicks();
    const uint64_t b = nowTicks();
    x = b - a;
  }
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  const size_t nOps = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1'000'000;
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(1, &set);
  if (sched_setaffinity(0, sizeof(set), &set) != 0) std::fprintf(stderr, "(無法綁核心，繼續)\n");
#endif
  const double tpn = ticksPerNs();
  const uint64_t overhead = measureTimerOverhead();
  std::printf("ops=%zu  計時器開銷≈%.1f ns（已扣除）  容量=%u\n\n", nOps, overhead / tpn, kCap);

  const Workload workloads[] = {Workload::Near, Workload::Mixed, Workload::DeepChurn, Workload::AggrChurn,
                                Workload::Requeue};
  for (Workload w : workloads) {
    std::printf("== %s：%s\n", workloadName(w), workloadDesc(w));
    std::printf("   L    impl   mean   p50    p99    p99.9  p99.99 (ns)\n");
    for (int L : {8, 32, 128}) {
      const auto ops = makeOps(w, nOps, L, 42);
      const Result de = run<SideIndex<Side::Buy, kCap, 8>>(ops, tpn, overhead);
      const Result bk = run<BackSortedIndex<Side::Buy, kCap, 8>>(ops, tpn, overhead);
      const Result mp = run<MapIndex>(ops, tpn, overhead);
      auto row = [&](const char* name, const Result& r) {
        std::printf("  %4d  %-5s %6.1f %6.1f %6.1f %6.1f %6.1f\n", L, name, r.mean, r.p50, r.p99, r.p999, r.p9999);
      };
      row("de", de);
      row("back", bk);
      row("map", mp);
    }
    std::printf("\n");
  }

  // MaxEmpty 的取捨：延遲回收越多，p99 越低（中間價位的新增 / 刪除不必 memmove），
  // 但累積的空價位一次壓縮時 p99.9 會變高。
  std::printf("== MaxEmpty 取捨（L=128）\n");
  std::printf("  workload  impl/MaxEmpty  mean   p50    p99    p99.9  p99.99 (ns)\n");
  for (Workload w : {Workload::Near, Workload::Mixed}) {
    const auto ops = makeOps(w, nOps, 128, 42);
    auto row = [&](const char* name, const Result& r) {
      std::printf("  %-9s %-13s %6.1f %6.1f %6.1f %6.1f %6.1f\n", workloadName(w), name, r.mean, r.p50, r.p99,
                  r.p999, r.p9999);
    };
    row("de/0", run<SideIndex<Side::Buy, kCap, 0>>(ops, tpn, overhead));
    row("de/8", run<SideIndex<Side::Buy, kCap, 8>>(ops, tpn, overhead));
    row("de/64", run<SideIndex<Side::Buy, kCap, 64>>(ops, tpn, overhead));
    row("back/0", run<BackSortedIndex<Side::Buy, kCap, 0>>(ops, tpn, overhead));
    row("back/8", run<BackSortedIndex<Side::Buy, kCap, 8>>(ops, tpn, overhead));
    row("back/64", run<BackSortedIndex<Side::Buy, kCap, 64>>(ops, tpn, overhead));
    row("map", run<MapIndex>(ops, tpn, overhead));
  }
  return 0;
}
