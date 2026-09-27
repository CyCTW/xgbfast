// 策略自有掛單的價位索引：雙端排序陣列 + 固定容量價位池。
//
// 設計重點：
//   * 價格為整數 tick（int64_t），不使用浮點數。
//   * 陣列從中間開始放，兩端都預留空間：
//       - 新的最佳價（積極回補）push 到尾端，O(1)
//       - 新的最差價（遠端被動單）push 到頭端，O(1)
//       - 中間插入時往比較近的一端 memmove
//   * 內部以 key 排序：買方 key = px，賣方 key = -px。
//     陣列依 key 遞增排列，最佳價永遠在尾端（key 最大）。
//   * PriceLevel 存在固定容量的 pool 中，levelId 不會因陣列搬移而改變，
//     訂單可以安全地持有 levelId。
//   * hot path 不配置記憶體、不丟例外；容量滿時回傳 kInvalid，由呼叫端決定是否拒單。
//   * 空價位延遲回收：價位的訂單數歸零時先保留（刪單後馬上在同價位重下很常見），
//     但維持不變式「陣列兩端一定是有掛單的價位」，所以 bestPx()/worstPx() 仍是 O(1)。
//     只有內部價位會處於「空但保留」狀態；空價位超過 MaxEmpty 個、或容量滿需要新價位時，
//     以一次 O(n) 壓縮回收。呼叫端也可以在空閒時間主動呼叫 purgeEmpty()。
#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

namespace oms {

enum class Side : uint8_t { Buy, Sell };

inline constexpr uint32_t kInvalid = UINT32_MAX;

struct PriceLevel {
  int64_t px = 0;
  int64_t totalQty = 0;          // 此價位所有訂單剩餘量總和
  uint32_t orderCount = 0;       // 此價位訂單數；0 表示空價位（可能處於延遲回收狀態）
  uint32_t head = kInvalid;      // FIFO 訂單鏈結（由 OrderSide 維護；只用 SideIndex 時不動）
  uint32_t tail = kInvalid;
  uint32_t nextFree = kInvalid;  // pool free list
};

// 固定容量價位池，free list 管理，O(1) 取得 / 歸還。
template <uint32_t Cap>
class LevelPool {
 public:
  LevelPool() {
    for (uint32_t i = 0; i < Cap; ++i) levels_[i].nextFree = (i + 1 < Cap) ? i + 1 : kInvalid;
    freeHead_ = 0;
  }

  uint32_t alloc(int64_t px) {
    const uint32_t id = freeHead_;
    if (id == kInvalid) return kInvalid;
    freeHead_ = levels_[id].nextFree;
    levels_[id] = PriceLevel{px, 0, 0, kInvalid, kInvalid, kInvalid};
    ++used_;
    return id;
  }

  void release(uint32_t id) {
    assert(id < Cap);
    levels_[id].nextFree = freeHead_;
    freeHead_ = id;
    --used_;
  }

  PriceLevel& operator[](uint32_t id) { return levels_[id]; }
  const PriceLevel& operator[](uint32_t id) const { return levels_[id]; }
  uint32_t used() const { return used_; }

 private:
  PriceLevel levels_[Cap];
  uint32_t freeHead_ = kInvalid;
  uint32_t used_ = 0;
};

// 單邊（買或賣）的價位索引。
//   Cap      : 最多可同時存在的價位數（含延遲回收中的空價位）
//   MaxEmpty : 最多保留幾個空價位；0 等同立即回收
template <Side S, uint32_t Cap = 256, uint32_t MaxEmpty = 8>
class SideIndex {
  static_assert(Cap >= 4, "capacity too small");
  static_assert(MaxEmpty < Cap, "MaxEmpty must be smaller than Cap");

 public:
  struct Ref {
    int64_t key;       // 買方 = px，賣方 = -px
    uint32_t levelId;  // 指向 pool
  };

  // 價位數不超過此值時用線性掃描，否則用二分搜尋。
  static constexpr uint32_t kLinearScanMax = 32;

  static constexpr int64_t toKey(int64_t px) { return S == Side::Buy ? px : -px; }
  static constexpr int64_t toPx(int64_t key) { return S == Side::Buy ? key : -key; }

  // 兩端一定非空，所以「沒有槽位」等同「沒有任何掛單價位」
  bool empty() const { return head_ == tail_; }
  uint32_t size() const { return slots() - emptyCount_; }  // 有掛單的價位數
  uint32_t slots() const { return tail_ - head_; }         // 含空價位的槽位數
  uint32_t emptyCount() const { return emptyCount_; }
  static constexpr uint32_t capacity() { return Cap; }

  // 呼叫前需確認 !empty()。不變式保證兩端都是有掛單的價位。
  int64_t bestPx() const { return toPx(buf_[tail_ - 1].key); }
  int64_t worstPx() const { return toPx(buf_[head_].key); }
  uint32_t bestLevel() const { return buf_[tail_ - 1].levelId; }

  // 精確查找，找不到或價位已空時回傳 kInvalid。
  uint32_t find(int64_t px) const {
    const int64_t key = toKey(px);
    const uint32_t pos = lowerBound(key);
    if (pos < tail_ && buf_[pos].key == key) {
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

  // 在 px 新增一張訂單（訂單數 +1、總量 +qty），回傳 levelId。
  // 容量滿（有掛單的價位數 == Cap）時回傳 kInvalid。
  uint32_t addOrder(int64_t px, int64_t qty) {
    const uint32_t id = acquireLevel(px);
    if (id == kInvalid) return kInvalid;
    pool_[id].totalQty += qty;
    pool_[id].orderCount += 1;
    return id;
  }

  // 訂單加量（不改變訂單數），例如改單加量。
  void addQtyById(uint32_t id, int64_t qty) { pool_[id].totalQty += qty; }

  // 訂單減量（成交或改量）。removeOrder=true 表示此訂單已離開此價位。
  // 訂單數歸零時價位進入空狀態：位於兩端就立即回收，否則延遲回收。
  // 呼叫後 id 可能已被回收，不可再使用。
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

  // 回收所有空價位（一次 O(slots) 壓縮）。適合在策略空閒時間呼叫。
  void purgeEmpty() {
    if (emptyCount_ == 0) return;
    uint32_t w = head_;
    for (uint32_t r = head_; r < tail_; ++r) {
      if (pool_[buf_[r].levelId].orderCount == 0) pool_.release(buf_[r].levelId);
      else buf_[w++] = buf_[r];
    }
    tail_ = w;
    emptyCount_ = 0;
    if (empty()) head_ = tail_ = Cap / 2;
  }

  // 由最佳價往最差價走訪。f(const PriceLevel&) 回傳 false 可提前結束。
  // 空價位會被跳過。
  template <class F>
  void forEachBestToWorst(F&& f) const {
    for (uint32_t i = tail_; i > head_; --i) {
      const PriceLevel& L = pool_[buf_[i - 1].levelId];
      if (L.orderCount > 0 && !f(L)) return;
    }
  }

  // 以下供測試檢查內部狀態
  uint32_t headPos() const { return head_; }
  uint32_t tailPos() const { return tail_; }
  uint32_t poolUsed() const { return pool_.used(); }
  bool checkInvariants() const {
    if (head_ > tail_ || tail_ > Cap) return false;
    if (pool_.used() != slots()) return false;
    if (emptyCount_ > MaxEmpty) return false;
    if (!empty() && (pool_[buf_[head_].levelId].orderCount == 0 ||
                     pool_[buf_[tail_ - 1].levelId].orderCount == 0))
      return false;  // 兩端必須非空
    uint32_t empties = 0;
    for (uint32_t i = head_; i < tail_; ++i) {
      if (i > head_ && !(buf_[i - 1].key < buf_[i].key)) return false;  // 嚴格遞增
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
  // 找到價位（含延遲回收中的空價位，會被重新啟用）或新建。容量滿時回傳 kInvalid。
  // 回傳時 orderCount 可能仍為 0，呼叫端必須立即加上訂單。
  uint32_t acquireLevel(int64_t px) {
    const int64_t key = toKey(px);

    // 快速路徑 1：空的，或新的最佳價（積極價）→ 放尾端
    if (empty() || key > buf_[tail_ - 1].key) {
      if (tail_ == Cap && !makeRoomBack()) return kInvalid;
      const uint32_t id = pool_.alloc(px);
      if (id == kInvalid) return kInvalid;
      buf_[tail_++] = Ref{key, id};
      return id;
    }
    // 快速路徑 2：新的最差價（遠端被動價）→ 放頭端
    if (key < buf_[head_].key) {
      if (head_ == 0 && !makeRoomFront()) return kInvalid;
      const uint32_t id = pool_.alloc(px);
      if (id == kInvalid) return kInvalid;
      buf_[--head_] = Ref{key, id};
      return id;
    }

    uint32_t pos = lowerBound(key);
    if (buf_[pos].key == key) {  // 已存在；若是空價位則重新啟用
      const uint32_t id = buf_[pos].levelId;
      if (pool_[id].orderCount == 0) --emptyCount_;
      return id;
    }

    // 中間插入：往比較近、且還有空間的一端搬
    const uint32_t leftCount = pos - head_;
    const uint32_t rightCount = tail_ - pos;
    bool shiftLeft = leftCount < rightCount;
    if (shiftLeft && head_ == 0) shiftLeft = false;
    if (!shiftLeft && tail_ == Cap) shiftLeft = true;
    if (shiftLeft && head_ == 0) {
      // 兩端都滿：（必要時先回收空價位）整段移回中間後重新計算位置
      if (!recenter(/*roomAtFront=*/false)) return kInvalid;
      pos = lowerBound(key);
      shiftLeft = (pos - head_) < (tail_ - pos);
      if (shiftLeft && head_ == 0) shiftLeft = false;  // 只剩一格空間時可能只有一端有空位
      if (!shiftLeft && tail_ == Cap) shiftLeft = true;
    }

    const uint32_t id = pool_.alloc(px);
    if (id == kInvalid) return kInvalid;
    if (shiftLeft) {
      // [head_, pos) 往左移一格，新元素放在 pos-1
      std::memmove(&buf_[head_ - 1], &buf_[head_], (pos - head_) * sizeof(Ref));
      --head_;
      buf_[pos - 1] = Ref{key, id};
    } else {
      // [pos, tail_) 往右移一格，新元素放在 pos
      std::memmove(&buf_[pos + 1], &buf_[pos], (tail_ - pos) * sizeof(Ref));
      ++tail_;
      buf_[pos] = Ref{key, id};
    }
    return id;
  }

  // 回收兩端的空價位，維持「兩端非空」不變式。
  void trimEnds() {
    while (head_ < tail_ && pool_[buf_[tail_ - 1].levelId].orderCount == 0) {
      pool_.release(buf_[--tail_].levelId);
      --emptyCount_;
    }
    while (head_ < tail_ && pool_[buf_[head_].levelId].orderCount == 0) {
      pool_.release(buf_[head_++].levelId);
      --emptyCount_;
    }
    if (empty()) head_ = tail_ = Cap / 2;  // 清空時回到中間
  }

  // 回傳第一個 key >= target 的位置（範圍 [head_, tail_]）。
  // 小陣列從最佳價端（尾端）往回線性掃描：常用價位都在尾端附近。
  uint32_t lowerBound(int64_t key) const {
    if (slots() <= kLinearScanMax) {
      uint32_t i = tail_;
      while (i > head_ && buf_[i - 1].key >= key) --i;
      return i;
    }
    uint32_t lo = head_, hi = tail_;
    while (lo < hi) {
      const uint32_t mid = lo + (hi - lo) / 2;
      if (buf_[mid].key < key) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  // 把資料整段移到中間，讓兩端各有空間。容量全滿時回傳 false。
  // 空位為奇數時，多出的一格給需要的那一端（roomAtFront 決定），
  // 保證只剩一格空間時也能在指定端插入。
  // 槽位全滿但有空價位時，先壓縮回收。
  bool recenter(bool roomAtFront) {
    if (slots() >= Cap) {
      if (emptyCount_ == 0) return false;
      purgeEmpty();
    }
    const uint32_t n = slots();
    const uint32_t spare = Cap - n;
    const uint32_t newHead = roomAtFront ? (spare + 1) / 2 : spare / 2;
    std::memmove(&buf_[newHead], &buf_[head_], n * sizeof(Ref));
    head_ = newHead;
    tail_ = newHead + n;
    return true;
  }
  bool makeRoomBack() { return recenter(false); }  // spare>=1 時保證 tail_ < Cap
  bool makeRoomFront() { return recenter(true); }  // spare>=1 時保證 head_ > 0

  Ref buf_[Cap];
  uint32_t head_ = Cap / 2;
  uint32_t tail_ = Cap / 2;
  uint32_t emptyCount_ = 0;  // 延遲回收中的空價位數（只會在內部，不會在兩端）
  LevelPool<Cap> pool_;
};

}  // namespace oms
