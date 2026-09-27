// 原方案一：單端排序陣列（只在尾端留空間），用來跟 SideIndex（雙端）比較。
//
//   * 資料固定放在 [0, n)，依 key 遞增排列，最佳價在尾端（買方 key = px，賣方 key = -px）。
//   * 新的最佳價：直接 push 到尾端，O(1)
//   * 新的最差價：要把整段往右搬一格，O(n)
//   * 中間插入 / 刪除：一律搬動插入點右邊的元素
//   * 最差端清空：要把整段往左搬
//
// 除了陣列配置之外，其餘行為（API、價位池、空價位延遲回收、兩端非空不變式、
// 線性掃描 / 二分搜尋切換點）都和 SideIndex 相同，因此可以直接互換做比較。
#pragma once

#include "side_index.hpp"

namespace oms {

template <Side S, uint32_t Cap = 256, uint32_t MaxEmpty = 8>
class BackSortedIndex {
  static_assert(Cap >= 4, "capacity too small");
  static_assert(MaxEmpty < Cap, "MaxEmpty must be smaller than Cap");

 public:
  struct Ref {
    int64_t key;
    uint32_t levelId;
  };

  static constexpr uint32_t kLinearScanMax = 32;

  static constexpr int64_t toKey(int64_t px) { return S == Side::Buy ? px : -px; }
  static constexpr int64_t toPx(int64_t key) { return S == Side::Buy ? key : -key; }

  bool empty() const { return n_ == 0; }
  uint32_t size() const { return n_ - emptyCount_; }
  uint32_t slots() const { return n_; }
  uint32_t emptyCount() const { return emptyCount_; }
  static constexpr uint32_t capacity() { return Cap; }

  int64_t bestPx() const { return toPx(buf_[n_ - 1].key); }
  int64_t worstPx() const { return toPx(buf_[0].key); }
  uint32_t bestLevel() const { return buf_[n_ - 1].levelId; }

  uint32_t find(int64_t px) const {
    const int64_t key = toKey(px);
    const uint32_t pos = lowerBound(key);
    if (pos < n_ && buf_[pos].key == key) {
      const uint32_t id = buf_[pos].levelId;
      if (pool_[id].orderCount > 0) return id;
    }
    return kInvalid;
  }

  PriceLevel* level(int64_t px) {
    const uint32_t id = find(px);
    return id == kInvalid ? nullptr : &pool_[id];
  }
  const PriceLevel& levelById(uint32_t id) const { return pool_[id]; }
  PriceLevel& levelById(uint32_t id) { return pool_[id]; }

  uint32_t addOrder(int64_t px, int64_t qty) {
    const uint32_t id = acquireLevel(px);
    if (id == kInvalid) return kInvalid;
    pool_[id].totalQty += qty;
    pool_[id].orderCount += 1;
    return id;
  }

  void addQtyById(uint32_t id, int64_t qty) { pool_[id].totalQty += qty; }

  void reduceById(uint32_t id, int64_t qty, bool removeOrder) {
    PriceLevel& L = pool_[id];
    assert(L.orderCount > 0 && L.totalQty >= qty);
    L.totalQty -= qty;
    if (removeOrder && --L.orderCount == 0) {
      ++emptyCount_;
      trimEnds();
      if (emptyCount_ > MaxEmpty) purgeEmpty();
    }
  }

  bool reduce(int64_t px, int64_t qty, bool removeOrder) {
    const uint32_t id = find(px);
    if (id == kInvalid) return false;
    reduceById(id, qty, removeOrder);
    return true;
  }

  void purgeEmpty() {
    if (emptyCount_ == 0) return;
    uint32_t w = 0;
    for (uint32_t r = 0; r < n_; ++r) {
      if (pool_[buf_[r].levelId].orderCount == 0) pool_.release(buf_[r].levelId);
      else buf_[w++] = buf_[r];
    }
    n_ = w;
    emptyCount_ = 0;
  }

  template <class F>
  void forEachBestToWorst(F&& f) const {
    for (uint32_t i = n_; i > 0; --i) {
      const PriceLevel& L = pool_[buf_[i - 1].levelId];
      if (L.orderCount > 0 && !f(L)) return;
    }
  }

  bool checkInvariants() const {
    if (n_ > Cap) return false;
    if (pool_.used() != n_) return false;
    if (emptyCount_ > MaxEmpty) return false;
    if (n_ > 0 && (pool_[buf_[0].levelId].orderCount == 0 || pool_[buf_[n_ - 1].levelId].orderCount == 0))
      return false;
    uint32_t empties = 0;
    for (uint32_t i = 0; i < n_; ++i) {
      if (i > 0 && !(buf_[i - 1].key < buf_[i].key)) return false;
      const PriceLevel& L = pool_[buf_[i].levelId];
      if (L.px != toPx(buf_[i].key)) return false;
      if (L.orderCount == 0) {
        ++empties;
        if (L.totalQty != 0) return false;
      }
    }
    return empties == emptyCount_;
  }

 private:
  uint32_t acquireLevel(int64_t px) {
    const int64_t key = toKey(px);

    // 新的最佳價：O(1)
    if (n_ == 0 || key > buf_[n_ - 1].key) {
      if (n_ == Cap && !makeRoom()) return kInvalid;
      const uint32_t id = pool_.alloc(px);
      if (id == kInvalid) return kInvalid;
      buf_[n_++] = Ref{key, id};
      return id;
    }

    uint32_t pos = lowerBound(key);
    if (buf_[pos].key == key) {
      const uint32_t id = buf_[pos].levelId;
      if (pool_[id].orderCount == 0) --emptyCount_;
      return id;
    }
    if (n_ == Cap) {
      if (!makeRoom()) return kInvalid;
      pos = lowerBound(key);
    }
    const uint32_t id = pool_.alloc(px);
    if (id == kInvalid) return kInvalid;
    // 其餘情況（包含新的最差價）：插入點右邊整段往右搬
    std::memmove(&buf_[pos + 1], &buf_[pos], (n_ - pos) * sizeof(Ref));
    buf_[pos] = Ref{key, id};
    ++n_;
    return id;
  }

  bool makeRoom() {
    if (emptyCount_ == 0) return false;
    purgeEmpty();
    return true;
  }

  void trimEnds() {
    while (n_ > 0 && pool_[buf_[n_ - 1].levelId].orderCount == 0) {
      pool_.release(buf_[--n_].levelId);
      --emptyCount_;
    }
    uint32_t k = 0;
    while (k < n_ && pool_[buf_[k].levelId].orderCount == 0) {
      pool_.release(buf_[k++].levelId);
      --emptyCount_;
    }
    if (k > 0) {  // 最差端清空：整段往左搬
      std::memmove(&buf_[0], &buf_[k], (n_ - k) * sizeof(Ref));
      n_ -= k;
    }
  }

  uint32_t lowerBound(int64_t key) const {
    if (n_ <= kLinearScanMax) {
      uint32_t i = n_;
      while (i > 0 && buf_[i - 1].key >= key) --i;
      return i;
    }
    uint32_t lo = 0, hi = n_;
    while (lo < hi) {
      const uint32_t mid = lo + (hi - lo) / 2;
      if (buf_[mid].key < key) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  Ref buf_[Cap];
  uint32_t n_ = 0;
  uint32_t emptyCount_ = 0;
  LevelPool<Cap> pool_;
};

}  // namespace oms
