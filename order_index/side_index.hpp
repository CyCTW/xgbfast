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
#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace oms {

enum class Side : uint8_t { Buy, Sell };

inline constexpr uint32_t kInvalid = UINT32_MAX;

struct PriceLevel {
  int64_t px = 0;
  int64_t totalQty = 0;     // 此價位所有訂單剩餘量總和
  uint32_t orderCount = 0;  // 此價位訂單數
  uint32_t nextFree = kInvalid;
  // 實務上這裡會再放 FIFO 訂單鏈結的 head/tail（侵入式雙向鏈結）
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
    levels_[id] = PriceLevel{px, 0, 0, kInvalid};
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

// 單邊（買或賣）的價位索引。Cap 為最多可同時存在的價位數。
template <Side S, uint32_t Cap = 256>
class SideIndex {
  static_assert(Cap >= 4, "capacity too small");

 public:
  struct Ref {
    int64_t key;       // 買方 = px，賣方 = -px
    uint32_t levelId;  // 指向 pool
  };

  // 價位數不超過此值時用線性掃描，否則用二分搜尋。
  static constexpr uint32_t kLinearScanMax = 32;

  static constexpr int64_t toKey(int64_t px) { return S == Side::Buy ? px : -px; }
  static constexpr int64_t toPx(int64_t key) { return S == Side::Buy ? key : -key; }

  bool empty() const { return head_ == tail_; }
  uint32_t size() const { return tail_ - head_; }
  static constexpr uint32_t capacity() { return Cap; }

  // 呼叫前需確認 !empty()
  int64_t bestPx() const { return toPx(buf_[tail_ - 1].key); }
  int64_t worstPx() const { return toPx(buf_[head_].key); }
  uint32_t bestLevel() const { return buf_[tail_ - 1].levelId; }

  // 精確查找，找不到回傳 kInvalid。
  uint32_t find(int64_t px) const {
    const int64_t key = toKey(px);
    const uint32_t pos = lowerBound(key);
    return (pos < tail_ && buf_[pos].key == key) ? buf_[pos].levelId : kInvalid;
  }

  PriceLevel* level(int64_t px) {
    const uint32_t id = find(px);
    return id == kInvalid ? nullptr : &pool_[id];
  }
  const PriceLevel& levelById(uint32_t id) const { return pool_[id]; }
  PriceLevel& levelById(uint32_t id) { return pool_[id]; }

  // 找到價位或新建。容量滿時回傳 kInvalid。
  uint32_t findOrInsert(int64_t px) {
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
    if (buf_[pos].key == key) return buf_[pos].levelId;  // 已存在

    // 中間插入：往比較近、且還有空間的一端搬
    const uint32_t leftCount = pos - head_;
    const uint32_t rightCount = tail_ - pos;
    bool shiftLeft = leftCount < rightCount;
    if (shiftLeft && head_ == 0) shiftLeft = false;
    if (!shiftLeft && tail_ == Cap) shiftLeft = true;
    if (shiftLeft && head_ == 0) {
      // 兩端都滿：整段移回中間後重新計算位置
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

  // 刪除價位（通常在 orderCount 歸零時呼叫）。價位不存在時回傳 false。
  bool erase(int64_t px) {
    const int64_t key = toKey(px);
    const uint32_t pos = lowerBound(key);
    if (pos >= tail_ || buf_[pos].key != key) return false;
    pool_.release(buf_[pos].levelId);

    if (pos == tail_ - 1) {  // 最佳價：O(1)
      --tail_;
    } else if (pos == head_) {  // 最差價：O(1)
      ++head_;
    } else if (pos - head_ < tail_ - pos) {  // 從左邊補位
      std::memmove(&buf_[head_ + 1], &buf_[head_], (pos - head_) * sizeof(Ref));
      ++head_;
    } else {  // 從右邊補位
      std::memmove(&buf_[pos], &buf_[pos + 1], (tail_ - pos - 1) * sizeof(Ref));
      --tail_;
    }
    if (empty()) head_ = tail_ = Cap / 2;  // 清空時回到中間
    return true;
  }

  // 加量 / 減量的便利函式：新增或移除一張訂單。
  // addOrder 容量滿時回傳 false。
  bool addOrder(int64_t px, int64_t qty) {
    const uint32_t id = findOrInsert(px);
    if (id == kInvalid) return false;
    pool_[id].totalQty += qty;
    pool_[id].orderCount += 1;
    return true;
  }

  // 訂單減量（成交或改量）。removeOrder=true 表示此訂單已離開此價位。
  // 價位訂單數歸零時自動刪除價位。
  bool reduce(int64_t px, int64_t qty, bool removeOrder) {
    const uint32_t id = find(px);
    if (id == kInvalid) return false;
    PriceLevel& L = pool_[id];
    assert(L.totalQty >= qty);
    L.totalQty -= qty;
    if (removeOrder) {
      assert(L.orderCount > 0);
      if (--L.orderCount == 0) erase(px);
    }
    return true;
  }

  // 由最佳價往最差價走訪。f(const PriceLevel&) 回傳 false 可提前結束。
  template <class F>
  void forEachBestToWorst(F&& f) const {
    for (uint32_t i = tail_; i > head_; --i)
      if (!f(pool_[buf_[i - 1].levelId])) return;
  }

  // 以下供測試檢查內部狀態
  uint32_t headPos() const { return head_; }
  uint32_t tailPos() const { return tail_; }
  uint32_t poolUsed() const { return pool_.used(); }
  bool checkInvariants() const {
    if (head_ > tail_ || tail_ > Cap) return false;
    if (pool_.used() != size()) return false;
    for (uint32_t i = head_; i < tail_; ++i) {
      if (i > head_ && !(buf_[i - 1].key < buf_[i].key)) return false;  // 嚴格遞增
      if (pool_[buf_[i].levelId].px != toPx(buf_[i].key)) return false;
    }
    return true;
  }

 private:
  // 回傳第一個 key >= target 的位置（範圍 [head_, tail_]）。
  // 小陣列從最佳價端（尾端）往回線性掃描：常用價位都在尾端附近。
  uint32_t lowerBound(int64_t key) const {
    if (size() <= kLinearScanMax) {
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
  bool recenter(bool roomAtFront) {
    const uint32_t n = size();
    if (n >= Cap) return false;
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
  LevelPool<Cap> pool_;
};

}  // namespace oms
