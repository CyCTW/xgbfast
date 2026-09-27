// OrderSide 的 fuzz 測試：以 std::map<px, std::vector<訂單>> 為參考實作，
// 驗證價位彙總、每個價位的 FIFO 排隊順序、排隊位置規則與延遲回收。
//
// 事件：
//   * 新單：90% mid ± 20 tick、5% 積極價、5% 遠端被動價
//   * 最佳價隊首成交（模擬市場成交）、隨機訂單部分成交
//   * 刪單（極端價位的單傾向很快被刪）
//   * 同價改量（減量保留位置、加量排到隊尾）
//   * 改價（含改到極端價位的回補）
//   * mid 跳空、空閒時主動回收空價位
//
// 用法：fuzz_order_side [iterations_per_seed] [num_seeds] [start_seed]
#include "back_sorted_index.hpp"
#include "order_side.hpp"

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

int g_failures = 0;

#define CHECK(cond, ...)                                                                   \
  do {                                                                                     \
    if (!(cond)) {                                                                         \
      std::fprintf(stderr, "CHECK failed: %s (%s:%d) ", #cond, __FILE__, __LINE__);       \
      std::fprintf(stderr, __VA_ARGS__);                                                   \
      std::fprintf(stderr, "\n");                                                          \
      ++g_failures;                                                                        \
      return false;                                                                        \
    }                                                                                      \
  } while (0)

struct RefOrder {
  uint32_t h;
  uint64_t clOrdId;
  int64_t leaves;
};

template <template <Side, uint32_t, uint32_t> class IndexT, Side S, uint32_t LevelCap, uint32_t OrderCap, uint32_t MaxEmpty>
class Harness {
 public:
  explicit Harness(uint64_t seed) : rng_(seed), seed_(seed) {}

  struct Stats {
    uint64_t adds = 0, fills = 0, partials = 0, cancels = 0, qtyDown = 0, qtyUp = 0, replaces = 0;
    uint64_t addRejects = 0, replaceRejects = 0, replaceViaFreedSlot = 0;
    uint32_t maxLevels = 0, maxOrders = 0;
  } stats_;

  bool run(uint64_t iterations) {
    for (it_ = 0; it_ < iterations; ++it_) {
      if (!step()) return false;
      if (!verify()) return false;
    }
    return true;
  }

 private:
  using Book = OrderSide<S, LevelCap, OrderCap, MaxEmpty, IndexT>;
  static constexpr bool kBuy = (S == Side::Buy);

  int64_t pick(int64_t lo, int64_t hi) { return std::uniform_int_distribution<int64_t>(lo, hi)(rng_); }
  bool chance(int pct) { return pick(0, 99) < pct; }
  int64_t aggressiveOffset(int64_t d) const { return kBuy ? d : -d; }

  // 依分布產生新價位；extreme 回報是否為極端價位
  int64_t genPx(bool& extreme) {
    const int k = (int)pick(0, 99);
    extreme = k >= 90;
    if (k < 90) return mid_ + pick(-20, 20);
    if (k < 95) return mid_ + aggressiveOffset(pick(200, 1'000'000));
    return mid_ - aggressiveOffset(pick(200, 1'000'000));
  }

  size_t refLevelCount() const { return ref_.size(); }

  std::vector<RefOrder>& refLevelOf(uint32_t h) { return ref_.at(pxOf_.at(h)); }
  size_t refPos(const std::vector<RefOrder>& q, uint32_t h) const {
    for (size_t i = 0; i < q.size(); ++i)
      if (q[i].h == h) return i;
    std::fprintf(stderr, "handle %u not found in ref level\n", h);
    std::exit(1);
  }

  void refErase(uint32_t h) {
    const int64_t px = pxOf_.at(h);
    auto& q = ref_.at(px);
    q.erase(q.begin() + (ptrdiff_t)refPos(q, h));
    if (q.empty()) ref_.erase(px);
    pxOf_.erase(h);
    live_.erase(std::find(live_.begin(), live_.end(), h));
  }

  // 挑一張存活訂單；有極端價位的單時高機率先挑它
  bool pickLive(uint32_t& h) {
    while (!extremes_.empty() && chance(50)) {
      const size_t i = (size_t)pick(0, (int64_t)extremes_.size() - 1);
      h = extremes_[i];
      extremes_[i] = extremes_.back();
      extremes_.pop_back();
      if (pxOf_.count(h)) return true;
    }
    if (live_.empty()) return false;
    h = live_[(size_t)pick(0, (int64_t)live_.size() - 1)];
    return true;
  }

  bool step() {
    const int op = (int)pick(0, 99);
    if (op < 2) {
      mid_ += pick(-5000, 5000);
      return true;
    }
    if (op < 4) {
      mid_ += pick(-3, 3);
      if (chance(30)) book_->purgeEmpty();
      return true;
    }

    const int addBias = live_.size() < OrderCap / 2 ? 40 : 25;
    if (op < 4 + addBias) return doAdd();

    uint32_t h;
    if (live_.empty()) return true;
    const int kind = (int)pick(0, 99);
    if (kind < 20) {  // 最佳價隊首成交
      h = book_->bestFront();
      auto& best = kBuy ? ref_.rbegin()->second : ref_.begin()->second;
      CHECK(best.front().h == h, "seed=%llu it=%llu bestFront mismatch", (unsigned long long)seed_, (unsigned long long)it_);
      return doFill(h);
    }
    if (!pickLive(h)) return true;
    if (kind < 35) return doFill(h);
    if (kind < 65) return doCancel(h);
    if (kind < 80) return doModifyQty(h);
    return doReplace(h);
  }

  bool doAdd() {
    bool extreme;
    const int64_t px = genPx(extreme);
    const int64_t qty = pick(1, 100);
    const uint64_t id = nextClOrdId_++;
    const uint32_t h = book_->add(id, px, qty);
    if (h == kInvalid) {
      const bool poolFull = live_.size() == OrderCap;
      const bool levelsFull = !ref_.count(px) && refLevelCount() == LevelCap;
      CHECK(poolFull || levelsFull, "seed=%llu it=%llu add rejected unexpectedly px=%lld", (unsigned long long)seed_, (unsigned long long)it_, (long long)px);
      ++stats_.addRejects;
      return true;
    }
    CHECK(!pxOf_.count(h), "seed=%llu it=%llu handle %u reused while live", (unsigned long long)seed_, (unsigned long long)it_, h);
    ref_[px].push_back({h, id, qty});
    pxOf_[h] = px;
    live_.push_back(h);
    if (extreme) extremes_.push_back(h);
    ++stats_.adds;
    return true;
  }

  bool doFill(uint32_t h) {
    auto& q = refLevelOf(h);
    RefOrder& r = q[refPos(q, h)];
    const int64_t qty = chance(50) ? r.leaves : pick(1, r.leaves);
    const bool alive = book_->reduce(h, qty);
    CHECK(alive == (qty < r.leaves), "seed=%llu it=%llu reduce return", (unsigned long long)seed_, (unsigned long long)it_);
    if (qty == r.leaves) {
      refErase(h);
      ++stats_.fills;
    } else {
      r.leaves -= qty;  // 保留排隊位置
      ++stats_.partials;
    }
    return true;
  }

  bool doCancel(uint32_t h) {
    book_->remove(h);
    refErase(h);
    ++stats_.cancels;
    return true;
  }

  bool doModifyQty(uint32_t h) {
    auto& q = refLevelOf(h);
    const size_t pos = refPos(q, h);
    const int64_t newQty = pick(1, 150);
    book_->modifyQty(h, newQty);
    if (newQty < q[pos].leaves) {
      q[pos].leaves = newQty;  // 減量：位置不變
      ++stats_.qtyDown;
    } else if (newQty > q[pos].leaves) {
      RefOrder r = q[pos];  // 加量：排到隊尾
      r.leaves = newQty;
      q.erase(q.begin() + (ptrdiff_t)pos);
      q.push_back(r);
      ++stats_.qtyUp;
    }
    return true;
  }

  bool doReplace(uint32_t h) {
    bool extreme;
    const int64_t newPx = genPx(extreme);
    const int64_t newQty = pick(1, 100);
    const int64_t oldPx = pxOf_.at(h);
    auto& oldQ = ref_.at(oldPx);
    const size_t pos = refPos(oldQ, h);
    const bool oldAlone = oldQ.size() == 1;
    const bool levelsFull = !ref_.count(newPx) && refLevelCount() == LevelCap;

    const bool ok = book_->replace(h, newPx, newQty);
    if (!ok) {
      CHECK(newPx != oldPx && levelsFull && !oldAlone, "seed=%llu it=%llu replace rejected unexpectedly", (unsigned long long)seed_, (unsigned long long)it_);
      ++stats_.replaceRejects;
      return true;
    }
    if (levelsFull && oldAlone && newPx != oldPx) ++stats_.replaceViaFreedSlot;

    if (newPx == oldPx) {  // 同價：依改量規則
      if (newQty < oldQ[pos].leaves) {
        oldQ[pos].leaves = newQty;
      } else if (newQty > oldQ[pos].leaves) {
        RefOrder r = oldQ[pos];
        r.leaves = newQty;
        oldQ.erase(oldQ.begin() + (ptrdiff_t)pos);
        oldQ.push_back(r);
      }
    } else {
      RefOrder r = oldQ[pos];
      r.leaves = newQty;
      oldQ.erase(oldQ.begin() + (ptrdiff_t)pos);
      if (oldQ.empty()) ref_.erase(oldPx);
      ref_[newPx].push_back(r);
      pxOf_[h] = newPx;
      if (extreme) extremes_.push_back(h);
    }
    ++stats_.replaces;
    return true;
  }

  bool verify() {
    const auto seed = (unsigned long long)seed_;
    const auto it = (unsigned long long)it_;
    CHECK(book_->checkInvariants(), "seed=%llu it=%llu internal invariants", seed, it);
    CHECK(book_->liveOrders() == live_.size(), "seed=%llu it=%llu live %u vs %zu", seed, it, book_->liveOrders(), live_.size());
    CHECK(book_->index().size() == ref_.size(), "seed=%llu it=%llu levels %u vs %zu", seed, it, book_->index().size(), ref_.size());
    stats_.maxLevels = std::max<uint32_t>(stats_.maxLevels, (uint32_t)ref_.size());
    stats_.maxOrders = std::max<uint32_t>(stats_.maxOrders, (uint32_t)live_.size());
    if (ref_.empty()) {
      CHECK(book_->empty(), "seed=%llu it=%llu expected empty", seed, it);
      return true;
    }
    const int64_t refBest = kBuy ? ref_.rbegin()->first : ref_.begin()->first;
    const int64_t refWorst = kBuy ? ref_.begin()->first : ref_.rbegin()->first;
    CHECK(book_->bestPx() == refBest, "seed=%llu it=%llu best %lld vs %lld", seed, it, (long long)book_->bestPx(), (long long)refBest);
    CHECK(book_->worstPx() == refWorst, "seed=%llu it=%llu worst", seed, it);

    // 價位彙總：最佳 → 最差
    std::vector<int64_t> pxs;
    for (auto& [px, q] : ref_) pxs.push_back(px);
    if (kBuy) std::reverse(pxs.begin(), pxs.end());
    size_t i = 0;
    bool match = true;
    book_->index().forEachBestToWorst([&](const PriceLevel& L) {
      if (i >= pxs.size() || L.px != pxs[i]) { match = false; return false; }
      int64_t sum = 0;
      for (auto& r : ref_.at(pxs[i])) sum += r.leaves;
      if (L.totalQty != sum || L.orderCount != ref_.at(pxs[i]).size()) { match = false; return false; }
      ++i;
      return true;
    });
    CHECK(match && i == pxs.size(), "seed=%llu it=%llu level mismatch at %zu", seed, it, i);

    // 每個價位的 FIFO 排隊順序
    for (auto& [px, q] : ref_) {
      size_t k = 0;
      bool fifoOk = true;
      book_->forEachOrderAt(px, [&](uint32_t h, const OrderNode& o) {
        if (k >= q.size() || q[k].h != h || q[k].leaves != o.leaves || q[k].clOrdId != o.clOrdId) {
          fifoOk = false;
          return false;
        }
        ++k;
        return true;
      });
      CHECK(fifoOk && k == q.size(), "seed=%llu it=%llu FIFO mismatch at px=%lld pos=%zu", seed, it, (long long)px, k);
    }

    // 排隊前方量
    for (int n = 0; n < 3; ++n) {
      const uint32_t h = live_[(size_t)pick(0, (int64_t)live_.size() - 1)];
      auto& q = refLevelOf(h);
      int64_t ahead = 0;
      for (size_t j = 0; j < refPos(q, h); ++j) ahead += q[j].leaves;
      CHECK(book_->qtyAhead(h) == ahead, "seed=%llu it=%llu qtyAhead", seed, it);
    }

    // 不存在的價位查不到訂單
    const int64_t probe = mid_ + pick(-2'000'000, 2'000'000);
    if (!ref_.count(probe)) {
      bool any = false;
      book_->forEachOrderAt(probe, [&](uint32_t, const OrderNode&) { any = true; return false; });
      CHECK(!any, "seed=%llu it=%llu phantom orders at %lld", seed, it, (long long)probe);
    }
    return true;
  }

  std::mt19937_64 rng_;
  uint64_t seed_;
  uint64_t it_ = 0;
  int64_t mid_ = 1'000'000;
  uint64_t nextClOrdId_ = 1;
  std::unique_ptr<Book> book_ = std::make_unique<Book>();
  std::map<int64_t, std::vector<RefOrder>> ref_;  // px → FIFO
  std::map<uint32_t, int64_t> pxOf_;              // handle → px
  std::vector<uint32_t> live_;
  std::vector<uint32_t> extremes_;
};

template <template <Side, uint32_t, uint32_t> class IndexT, Side S, uint32_t LevelCap, uint32_t OrderCap, uint32_t MaxEmpty>
bool runConfig(const char* name, uint64_t iters, uint64_t seeds, uint64_t startSeed) {
  typename Harness<IndexT, S, LevelCap, OrderCap, MaxEmpty>::Stats t{};
  for (uint64_t s = startSeed; s < startSeed + seeds; ++s) {
    auto h = std::make_unique<Harness<IndexT, S, LevelCap, OrderCap, MaxEmpty>>(s);
    if (!h->run(iters)) {
      std::fprintf(stderr, "[%s] FAILED at seed %llu\n", name, (unsigned long long)s);
      return false;
    }
    const auto& x = h->stats_;
    t.adds += x.adds; t.fills += x.fills; t.partials += x.partials; t.cancels += x.cancels;
    t.qtyDown += x.qtyDown; t.qtyUp += x.qtyUp; t.replaces += x.replaces;
    t.addRejects += x.addRejects; t.replaceRejects += x.replaceRejects;
    t.replaceViaFreedSlot += x.replaceViaFreedSlot;
    t.maxLevels = std::max(t.maxLevels, x.maxLevels);
    t.maxOrders = std::max(t.maxOrders, x.maxOrders);
  }
  std::printf("[%-21s] ok  adds=%llu fills=%llu partial=%llu cancel=%llu qty-/+=%llu/%llu replace=%llu "
              "rejects(add=%llu repl=%llu) repl-freed-slot=%llu maxLevels=%u maxOrders=%u\n",
              name, (unsigned long long)t.adds, (unsigned long long)t.fills, (unsigned long long)t.partials,
              (unsigned long long)t.cancels, (unsigned long long)t.qtyDown, (unsigned long long)t.qtyUp,
              (unsigned long long)t.replaces, (unsigned long long)t.addRejects,
              (unsigned long long)t.replaceRejects, (unsigned long long)t.replaceViaFreedSlot, t.maxLevels,
              t.maxOrders);
  return true;
}

template <template <Side, uint32_t, uint32_t> class IndexT>
bool directedTests(const char* name) {
  // 1. FIFO 與排隊位置規則
  {
    OrderSide<Side::Buy, 16, 64, 4, IndexT> b;
    const uint32_t a = b.add(1, 100, 10);
    const uint32_t c = b.add(2, 100, 20);
    const uint32_t d = b.add(3, 100, 30);
    CHECK(b.qtyAhead(a) == 0 && b.qtyAhead(c) == 10 && b.qtyAhead(d) == 30, "initial queue");
    b.modifyQty(a, 5);  // 減量保留位置
    CHECK(b.qtyAhead(c) == 5 && b.bestFront() == a, "qty down keeps priority");
    b.modifyQty(a, 50);  // 加量排到隊尾
    CHECK(b.bestFront() == c && b.qtyAhead(a) == 50, "qty up loses priority");
    CHECK(b.reduce(c, 5) && b.bestFront() == c, "partial fill keeps priority");
    CHECK(!b.reduce(c, 15) && b.bestFront() == d, "full fill removes");
    CHECK(b.checkInvariants(), "invariants");
  }
  // 2. 改價到極端價位（積極回補）：handle 不變、成為新的最佳價
  {
    OrderSide<Side::Buy, 16, 64, 4, IndexT> b;
    const uint32_t a = b.add(1, 100, 10);
    b.add(2, 99, 10);
    CHECK(b.replace(a, 1'000'000, 3), "replace to aggressive");
    CHECK(b.bestPx() == 1'000'000 && b.bestFront() == a && b.order(a).leaves == 3, "aggressive best");
    CHECK(b.index().find(100) == kInvalid, "old level gone");
    CHECK(b.checkInvariants(), "invariants");
  }
  // 3. 價位容量全滿時改價：舊價位只剩這張單 → 成功；否則拒絕且狀態不變
  {
    OrderSide<Side::Sell, 4, 64, 2, IndexT> a;
    const uint32_t h1 = a.add(1, 100, 1);
    const uint32_t h2 = a.add(2, 101, 1);
    a.add(3, 101, 1);
    a.add(4, 102, 1);
    a.add(5, 103, 1);
    CHECK(a.add(6, 500, 1) == kInvalid, "levels full rejects add");
    CHECK(a.replace(h1, 500, 1), "alone on level → uses freed slot");
    CHECK(a.worstPx() == 500 && a.bestPx() == 101, "moved");
    CHECK(!a.replace(h2, 600, 1), "level shared → reject");
    CHECK(a.order(h2).px == 101 && a.checkInvariants(), "unchanged after reject");
  }
  // 4. 訂單池滿
  {
    OrderSide<Side::Buy, 8, 4, 2, IndexT> b;
    for (int i = 0; i < 4; ++i) CHECK(b.add(i, 100 + i, 1) != kInvalid, "fill pool");
    CHECK(b.add(9, 100, 1) == kInvalid, "pool full rejects");
    CHECK(b.checkInvariants(), "invariants");
  }
  std::printf("[directed %-12s] ok\n", name);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const uint64_t iters = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 20000;
  const uint64_t seeds = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 30;
  const uint64_t start = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 1;

  bool ok = directedTests<SideIndex>("de") && directedTests<BackSortedIndex>("back");
  // 價位與訂單容量都很小：頻繁撞到兩種容量上限
  ok = ok && runConfig<SideIndex, Side::Buy, 8, 24, 0>("de/buy/L8/O24/e0", iters, seeds, start);
  ok = ok && runConfig<SideIndex, Side::Sell, 8, 24, 3>("de/sell/L8/O24/e3", iters, seeds, start);
  // 訂單池是瓶頸
  ok = ok && runConfig<SideIndex, Side::Sell, 64, 12, 4>("de/sell/L64/O12/e4", iters, seeds, start);
  // 價位容量是瓶頸（訂單多、價位少）
  ok = ok && runConfig<SideIndex, Side::Buy, 16, 256, 4>("de/buy/L16/O256/e4", iters, seeds, start);
  // 一般容量
  ok = ok && runConfig<SideIndex, Side::Sell, 64, 128, 8>("de/sell/L64/O128/e8", iters, seeds, start);
  ok = ok && runConfig<SideIndex, Side::Buy, 256, 512, 16>("de/buy/L256/O512", iters, seeds, start);
  // 原方案一（單端排序陣列），同樣的組態
  ok = ok && runConfig<BackSortedIndex, Side::Buy, 8, 24, 0>("back/buy/L8/O24/e0", iters, seeds, start);
  ok = ok && runConfig<BackSortedIndex, Side::Sell, 8, 24, 3>("back/sell/L8/O24/e3", iters, seeds, start);
  ok = ok && runConfig<BackSortedIndex, Side::Sell, 64, 12, 4>("back/sell/L64/O12/e4", iters, seeds, start);
  ok = ok && runConfig<BackSortedIndex, Side::Buy, 16, 256, 4>("back/buy/L16/O256/e4", iters, seeds, start);
  ok = ok && runConfig<BackSortedIndex, Side::Sell, 64, 128, 8>("back/sell/L64/O128/e8", iters, seeds, start);
  ok = ok && runConfig<BackSortedIndex, Side::Buy, 256, 512, 16>("back/buy/L256/O512", iters, seeds, start);

  if (!ok || g_failures) {
    std::printf("FAILED (%d failures)\n", g_failures);
    return 1;
  }
  std::printf("ALL PASSED\n");
  return 0;
}
