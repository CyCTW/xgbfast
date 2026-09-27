// SideIndex 的 fuzz 測試：以 std::map 為參考實作，隨機事件後逐一比對。
//
// 事件分布（模擬「大部分價位接近市價，偶爾回補到極端價位」）：
//   * 約 90% 新增：mid ± 20 tick
//   * 約 5%  新增：積極價（買方遠高於 mid / 賣方遠低於 mid）
//   * 約 5%  新增：遠端被動價（反方向遠離 mid）
//   * 減量、成交、刪單：隨機挑既有價位；極端價位傾向很快被移除
//   * 偶爾 mid 跳空大幅移動
//
// 用法：fuzz_side_index [iterations_per_seed] [num_seeds] [start_seed]
#include "side_index.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <map>
#include <memory>
#include <random>
#include <vector>

using namespace oms;

namespace {

struct RefLevel {
  int64_t qty = 0;
  uint32_t count = 0;
};

int g_failures = 0;

#define CHECK(cond, ...)                                                  \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::fprintf(stderr, "CHECK failed: %s (%s:%d) ", #cond, __FILE__, \
                   __LINE__);                                             \
      std::fprintf(stderr, __VA_ARGS__);                                  \
      std::fprintf(stderr, "\n");                                         \
      ++g_failures;                                                       \
      return false;                                                       \
    }                                                                     \
  } while (0)

template <Side S, uint32_t Cap, uint32_t MaxEmpty>
class Harness {
 public:
  explicit Harness(uint64_t seed) : rng_(seed), seed_(seed) {}

  bool run(uint64_t iterations) {
    for (uint64_t it = 0; it < iterations; ++it) {
      step();
      if (!verify(it)) return false;
    }
    stats_.maxSize = maxSize_;
    return true;
  }

  struct Stats {
    uint64_t adds = 0, aggressive = 0, deepPassive = 0, reduces = 0, removes = 0;
    uint64_t full = 0, jumps = 0;
    uint32_t maxSize = 0;
  } stats_;

 private:
  using Index = SideIndex<S, Cap, MaxEmpty>;
  static constexpr bool kBuy = (S == Side::Buy);

  // 買方的積極價在上方，賣方的積極價在下方
  int64_t aggressiveOffset(int64_t d) const { return kBuy ? d : -d; }

  int64_t pick(int64_t lo, int64_t hi) { return std::uniform_int_distribution<int64_t>(lo, hi)(rng_); }
  bool chance(int pct) { return pick(0, 99) < pct; }

  void addAt(int64_t px, bool extreme) {
    const int64_t qty = pick(1, 100);
    const bool refHas = ref_.count(px) != 0;
    const bool ok = idx_.addOrder(px, qty) != kInvalid;
    if (!ok) {
      // 只有在價位不存在且容量已滿時才允許失敗
      if (refHas || ref_.size() < Cap) {
        std::fprintf(stderr, "addOrder failed unexpectedly px=%lld size=%zu\n", (long long)px, ref_.size());
        ++g_failures;
        std::exit(1);
      }
      ++stats_.full;
      return;
    }
    RefLevel& r = ref_[px];
    r.qty += qty;
    r.count += 1;
    if (extreme) extremes_.push_back(px);
    ++stats_.adds;
  }

  // 從參考實作中挑一個既有價位；有極端價位時高機率先挑它（模擬短暫存在）
  bool pickExisting(int64_t& px) {
    while (!extremes_.empty() && chance(60)) {
      const size_t i = (size_t)pick(0, (int64_t)extremes_.size() - 1);
      px = extremes_[i];
      extremes_[i] = extremes_.back();
      extremes_.pop_back();
      if (ref_.count(px)) return true;
    }
    if (ref_.empty()) return false;
    auto it = ref_.begin();
    std::advance(it, pick(0, (int64_t)ref_.size() - 1));
    px = it->first;
    return true;
  }

  void step() {
    const int op = (int)pick(0, 99);

    if (op < 2) {  // mid 跳空
      mid_ += pick(-5000, 5000);
      ++stats_.jumps;
      return;
    }
    if (op < 3) {  // 小幅漂移，順便模擬空閒時間主動回收空價位
      mid_ += pick(-3, 3);
      if (chance(20)) idx_.purgeEmpty();
      return;
    }

    // 價位數越多越傾向移除，避免長期塞滿；但仍會偶爾撞到容量上限
    const int addBias = ref_.size() < Cap / 2 ? 60 : (ref_.size() < Cap ? 45 : 30);
    if (op < 3 + addBias) {
      const int kind = (int)pick(0, 99);
      if (kind < 90) {
        addAt(mid_ + pick(-20, 20), false);
      } else if (kind < 95) {
        addAt(mid_ + aggressiveOffset(pick(200, 1'000'000)), true);
        ++stats_.aggressive;
      } else {
        addAt(mid_ - aggressiveOffset(pick(200, 1'000'000)), true);
        ++stats_.deepPassive;
      }
      return;
    }

    int64_t px;
    if (!pickExisting(px)) return;
    RefLevel& r = ref_[px];
    // 以「平均每張單的量」模擬單張訂單減量；整張離開時扣掉整張估計量
    const int64_t perOrder = r.qty / r.count;
    if (chance(50) && perOrder > 1) {  // 部分成交或減量，訂單還在
      const int64_t q = pick(1, perOrder - 1);
      const bool ok = idx_.reduce(px, q, false);
      if (!ok) { std::fprintf(stderr, "reduce failed px=%lld\n", (long long)px); ++g_failures; std::exit(1); }
      r.qty -= q;
      ++stats_.reduces;
    } else {  // 整張成交或刪單
      const int64_t q = (r.count == 1) ? r.qty : perOrder;
      const bool ok = idx_.reduce(px, q, true);
      if (!ok) { std::fprintf(stderr, "remove failed px=%lld\n", (long long)px); ++g_failures; std::exit(1); }
      r.qty -= q;
      if (--r.count == 0) ref_.erase(px);
      ++stats_.removes;
    }
  }

  bool verify(uint64_t it) {
    CHECK(idx_.checkInvariants(), "seed=%llu it=%llu internal invariants", (unsigned long long)seed_, (unsigned long long)it);
    CHECK(idx_.size() == ref_.size(), "seed=%llu it=%llu size %u vs %zu", (unsigned long long)seed_, (unsigned long long)it, idx_.size(), ref_.size());
    if (idx_.size() > maxSize_) maxSize_ = idx_.size();
    if (ref_.empty()) {
      CHECK(idx_.empty(), "seed=%llu it=%llu expected empty", (unsigned long long)seed_, (unsigned long long)it);
      return true;
    }

    const int64_t refBest = kBuy ? ref_.rbegin()->first : ref_.begin()->first;
    const int64_t refWorst = kBuy ? ref_.begin()->first : ref_.rbegin()->first;
    CHECK(idx_.bestPx() == refBest, "seed=%llu it=%llu best %lld vs %lld", (unsigned long long)seed_, (unsigned long long)it, (long long)idx_.bestPx(), (long long)refBest);
    CHECK(idx_.worstPx() == refWorst, "seed=%llu it=%llu worst", (unsigned long long)seed_, (unsigned long long)it);

    // 完整走訪比對（最佳 → 最差）
    std::vector<std::pair<int64_t, RefLevel>> expected(ref_.begin(), ref_.end());
    if (kBuy) std::reverse(expected.begin(), expected.end());
    size_t i = 0;
    bool match = true;
    idx_.forEachBestToWorst([&](const PriceLevel& L) {
      if (i >= expected.size() || L.px != expected[i].first || L.totalQty != expected[i].second.qty ||
          L.orderCount != expected[i].second.count) {
        match = false;
        return false;
      }
      ++i;
      return true;
    });
    CHECK(match && i == expected.size(), "seed=%llu it=%llu traversal mismatch at %zu", (unsigned long long)seed_, (unsigned long long)it, i);

    // 精確查找：抽幾個存在的價位、幾個不存在的價位
    for (int k = 0; k < 4; ++k) {
      auto itr = ref_.begin();
      std::advance(itr, pick(0, (int64_t)ref_.size() - 1));
      const uint32_t id = idx_.find(itr->first);
      CHECK(id != kInvalid && idx_.levelById(id).px == itr->first, "seed=%llu it=%llu find existing %lld", (unsigned long long)seed_, (unsigned long long)it, (long long)itr->first);
      const int64_t probe = mid_ + pick(-2'000'000, 2'000'000);
      CHECK((idx_.find(probe) != kInvalid) == (ref_.count(probe) != 0), "seed=%llu it=%llu find probe %lld", (unsigned long long)seed_, (unsigned long long)it, (long long)probe);
    }
    return true;
  }

  std::mt19937_64 rng_;
  uint64_t seed_;
  int64_t mid_ = 1'000'000;
  Index idx_;
  std::map<int64_t, RefLevel> ref_;
  std::vector<int64_t> extremes_;
  uint32_t maxSize_ = 0;
};

template <Side S, uint32_t Cap, uint32_t MaxEmpty>
bool runConfig(const char* name, uint64_t iters, uint64_t seeds, uint64_t startSeed) {
  typename Harness<S, Cap, MaxEmpty>::Stats total{};
  for (uint64_t s = startSeed; s < startSeed + seeds; ++s) {
    auto h = std::make_unique<Harness<S, Cap, MaxEmpty>>(s);  // 物件較大，放 heap
    if (!h->run(iters)) {
      std::fprintf(stderr, "[%s] FAILED at seed %llu\n", name, (unsigned long long)s);
      return false;
    }
    total.adds += h->stats_.adds;
    total.aggressive += h->stats_.aggressive;
    total.deepPassive += h->stats_.deepPassive;
    total.reduces += h->stats_.reduces;
    total.removes += h->stats_.removes;
    total.full += h->stats_.full;
    total.jumps += h->stats_.jumps;
    if (h->stats_.maxSize > total.maxSize) total.maxSize = h->stats_.maxSize;
  }
  std::printf("[%-12s] ok  adds=%llu (aggr=%llu deep=%llu) reduces=%llu removes=%llu full-rejects=%llu jumps=%llu maxLevels=%u\n",
              name, (unsigned long long)total.adds, (unsigned long long)total.aggressive,
              (unsigned long long)total.deepPassive, (unsigned long long)total.reduces,
              (unsigned long long)total.removes, (unsigned long long)total.full,
              (unsigned long long)total.jumps, total.maxSize);
  return true;
}

// 針對邊界情況的確定性測試
bool directedTests() {
  // 1. 積極價應立即成為最佳價；刪掉後最佳價恢復
  {
    SideIndex<Side::Buy, 16> b;
    for (int64_t p = 100; p < 105; ++p) b.addOrder(p, 1);
    b.addOrder(1'000'000, 5);  // 積極回補
    CHECK(b.bestPx() == 1'000'000, "aggressive best");
    b.addOrder(-1'000'000, 5);  // 遠端被動
    CHECK(b.worstPx() == -1'000'000, "deep worst");
    b.reduce(1'000'000, 5, true);
    CHECK(b.bestPx() == 104, "best restored");
    CHECK(b.checkInvariants(), "invariants");
  }
  // 2. 單邊持續 push（只往最佳端加）要能觸發 recenter 並填滿整個容量
  {
    SideIndex<Side::Sell, 8, 2> a;
    for (int64_t p = 100; p > 92; --p) CHECK(a.addOrder(p, 1) != kInvalid, "sell push best p=%lld", (long long)p);
    CHECK(a.size() == 8 && a.bestPx() == 93 && a.worstPx() == 100, "sell full");
    CHECK(a.addOrder(50, 1) == kInvalid, "full must reject new level");
    CHECK(a.addOrder(95, 1) != kInvalid, "existing level still ok when full");
    CHECK(a.checkInvariants(), "invariants");
  }
  // 3. 只往最差端加
  {
    SideIndex<Side::Buy, 8, 2> b;
    for (int64_t p = 100; p > 92; --p) CHECK(b.addOrder(p, 1) != kInvalid, "buy push worst p=%lld", (long long)p);
    CHECK(b.bestPx() == 100 && b.worstPx() == 93, "buy worst order");
    CHECK(b.checkInvariants(), "invariants");
  }
  // 4. 清空後回到中間
  {
    SideIndex<Side::Buy, 8, 2> b;
    b.addOrder(10, 1);
    b.reduce(10, 1, true);
    CHECK(b.empty() && b.headPos() == 4 && b.tailPos() == 4, "reset to center");
  }
  // 5. 延遲回收：內部價位清空時保留，兩端清空時連同相鄰空價位一起回收
  {
    SideIndex<Side::Buy, 16, 8> b;
    for (int64_t p = 100; p <= 104; ++p) b.addOrder(p, 1);
    b.reduce(101, 1, true);
    b.reduce(102, 1, true);
    b.reduce(103, 1, true);
    CHECK(b.size() == 2 && b.slots() == 5 && b.emptyCount() == 3, "interior levels kept");
    CHECK(b.find(102) == kInvalid && b.level(102) == nullptr, "empty level hidden from find");
    CHECK(b.bestPx() == 104, "best unaffected");
    const uint32_t id102 = b.addOrder(102, 7);  // 重新啟用
    CHECK(id102 != kInvalid && b.emptyCount() == 2 && b.find(102) == id102, "revive");
    b.reduce(104, 1, true);  // 最佳價清空 → 103 也是空的，一起回收
    CHECK(b.bestPx() == 102 && b.slots() == 3 && b.emptyCount() == 1, "trim best end");
    b.reduce(100, 1, true);  // 最差價清空 → 101 一起回收
    CHECK(b.worstPx() == 102 && b.bestPx() == 102 && b.slots() == 1 && b.emptyCount() == 0, "trim worst end");
    CHECK(b.checkInvariants(), "invariants");
  }
  // 6. 超過 MaxEmpty 時自動壓縮
  {
    SideIndex<Side::Sell, 16, 2> a;
    for (int64_t p = 100; p <= 106; ++p) a.addOrder(p, 1);
    a.reduce(101, 1, true);
    a.reduce(102, 1, true);
    CHECK(a.emptyCount() == 2 && a.slots() == 7, "within MaxEmpty");
    a.reduce(103, 1, true);
    CHECK(a.emptyCount() == 0 && a.slots() == 4, "purged");
    CHECK(a.checkInvariants(), "invariants");
  }
  // 7. 容量滿但有空價位時，新價位仍可插入（先壓縮）
  {
    SideIndex<Side::Buy, 8, 4> b;
    for (int64_t p = 100; p < 108; ++p) b.addOrder(p, 1);
    b.reduce(103, 1, true);
    CHECK(b.slots() == 8 && b.size() == 7, "full with one empty");
    CHECK(b.addOrder(200, 1) != kInvalid && b.bestPx() == 200, "insert best when full");
    b.reduce(104, 1, true);
    CHECK(b.addOrder(50, 1) != kInvalid && b.worstPx() == 50, "insert worst when full");
    b.reduce(105, 1, true);
    CHECK(b.addOrder(103, 1) != kInvalid && b.find(103) != kInvalid, "insert middle when full");
    CHECK(b.addOrder(104, 1) == kInvalid, "truly full rejects");
    CHECK(b.checkInvariants(), "invariants");
  }
  std::printf("[directed    ] ok\n");
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const uint64_t iters = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 20000;
  const uint64_t seeds = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 50;
  const uint64_t start = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 1;

  bool ok = directedTests();
  // 小容量：頻繁撞到兩端與容量上限，測 recenter 與拒單路徑
  ok = ok && runConfig<Side::Buy, 8, 0>("buy/8/e0", iters, seeds, start);
  ok = ok && runConfig<Side::Buy, 8, 3>("buy/8/e3", iters, seeds, start);
  ok = ok && runConfig<Side::Sell, 8, 3>("sell/8/e3", iters, seeds, start);
  // 中容量：線性掃描與二分搜尋切換點附近（kLinearScanMax = 32）
  ok = ok && runConfig<Side::Buy, 48, 8>("buy/48/e8", iters, seeds, start);
  ok = ok && runConfig<Side::Sell, 48, 8>("sell/48/e8", iters, seeds, start);
  // 正式容量
  ok = ok && runConfig<Side::Buy, 256, 8>("buy/256/e8", iters, seeds, start);
  ok = ok && runConfig<Side::Sell, 256, 32>("sell/256/e32", iters, seeds, start);

  if (!ok || g_failures) {
    std::printf("FAILED (%d failures)\n", g_failures);
    return 1;
  }
  std::printf("ALL PASSED\n");
  return 0;
}
