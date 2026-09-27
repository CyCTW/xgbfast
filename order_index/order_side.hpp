// 單邊自有訂單簿：SideIndex（價位索引）+ 訂單池 + 每個價位的 FIFO 訂單佇列。
//
// 設計重點：
//   * 訂單存在固定容量的池中，handle（池的索引）在訂單存活期間不變，
//     呼叫端可以把 handle 編進 clOrdId，回報進來時 O(1) 找到訂單。
//   * 每個價位的訂單以侵入式雙向鏈結串成 FIFO（head/tail 存在 PriceLevel），
//     依交易所排隊順序排列：新單接在隊尾。
//   * 排隊優先權規則（多數交易所）：
//       - 成交、減量：保留原本的排隊位置
//       - 加量：排到隊尾
//       - 改價：移到新價位的隊尾
//   * hot path 不配置記憶體；容量不足時回傳 kInvalid / false，不改變任何狀態。
#pragma once

#include "side_index.hpp"

namespace oms {

struct OrderNode {
  uint64_t clOrdId = 0;
  int64_t px = 0;
  int64_t leaves = 0;           // 剩餘量
  uint32_t levelId = kInvalid;  // 所在價位（SideIndex 的 levelId）
  uint32_t prev = kInvalid;     // 同價位 FIFO 的前一張（較早排隊）
  uint32_t next = kInvalid;     // 同價位 FIFO 的下一張；訂單釋放後作為 free list
  bool live = false;
};

// IndexT：價位索引的實作，預設 SideIndex（雙端），也可換成 BackSortedIndex（單端）等
// 具有相同 API 的類別。
template <Side S, uint32_t LevelCap = 256, uint32_t OrderCap = 4096, uint32_t MaxEmpty = 8,
          template <Side, uint32_t, uint32_t> class IndexT = SideIndex>
class OrderSide {
 public:
  using Index = IndexT<S, LevelCap, MaxEmpty>;

  OrderSide() {
    for (uint32_t i = 0; i < OrderCap; ++i) orders_[i].next = (i + 1 < OrderCap) ? i + 1 : kInvalid;
    freeHead_ = 0;
  }

  // 新增訂單並排到該價位隊尾，回傳 handle。訂單池或價位容量不足時回傳 kInvalid。
  uint32_t add(uint64_t clOrdId, int64_t px, int64_t qty) {
    assert(qty > 0);
    if (freeHead_ == kInvalid) return kInvalid;
    const uint32_t lid = idx_.addOrder(px, qty);
    if (lid == kInvalid) return kInvalid;
    const uint32_t h = freeHead_;
    freeHead_ = orders_[h].next;
    orders_[h] = OrderNode{clOrdId, px, qty, lid, kInvalid, kInvalid, true};
    linkBack(h, lid);
    ++live_;
    return h;
  }

  // 成交或減量 qty（保留排隊位置）。剩餘量歸零時訂單移除。回傳訂單是否仍存在。
  bool reduce(uint32_t h, int64_t qty) {
    OrderNode& o = orders_[h];
    assert(o.live && qty > 0 && qty <= o.leaves);
    if (qty == o.leaves) {
      remove(h);
      return false;
    }
    o.leaves -= qty;
    idx_.reduceById(o.levelId, qty, /*removeOrder=*/false);
    return true;
  }

  // 刪單、全部成交或被交易所取消。
  void remove(uint32_t h) {
    OrderNode& o = orders_[h];
    assert(o.live);
    unlink(h);
    idx_.reduceById(o.levelId, o.leaves, /*removeOrder=*/true);
    o.live = false;
    o.next = freeHead_;
    freeHead_ = h;
    --live_;
  }

  // 同價改量：減量保留排隊位置，加量排到隊尾。
  void modifyQty(uint32_t h, int64_t newQty) {
    OrderNode& o = orders_[h];
    assert(o.live && newQty > 0);
    if (newQty < o.leaves) {
      reduce(h, o.leaves - newQty);
    } else if (newQty > o.leaves) {
      idx_.addQtyById(o.levelId, newQty - o.leaves);
      o.leaves = newQty;
      unlink(h);
      linkBack(h, o.levelId);
    }
  }

  // 改價（可同時改量）：移到新價位隊尾，handle 不變。
  // 新價位無法建立時回傳 false，訂單維持原狀。
  bool replace(uint32_t h, int64_t newPx, int64_t newQty) {
    OrderNode& o = orders_[h];
    assert(o.live && newQty > 0);
    if (newPx == o.px) {
      modifyQty(h, newQty);
      return true;
    }
    const uint32_t oldLid = o.levelId;
    // 先建立新價位再離開舊價位：新價位建不起來時可以不動任何狀態。
    uint32_t newLid = idx_.addOrder(newPx, newQty);
    if (newLid == kInvalid) {
      // 價位容量滿，但如果舊價位只剩這張單，離開後就會空出一格。
      if (idx_.levelById(oldLid).orderCount != 1) return false;
      unlink(h);
      idx_.reduceById(oldLid, o.leaves, /*removeOrder=*/true);
      newLid = idx_.addOrder(newPx, newQty);
      assert(newLid != kInvalid);  // 舊價位已空（被回收或可被壓縮），一定有空間
    } else {
      unlink(h);
      idx_.reduceById(oldLid, o.leaves, /*removeOrder=*/true);
    }
    o.px = newPx;
    o.leaves = newQty;
    o.levelId = newLid;
    linkBack(h, newLid);
    return true;
  }

  // 排在這張單前面（同價位、較早排隊）的剩餘量總和。O(排隊位置)。
  int64_t qtyAhead(uint32_t h) const {
    int64_t sum = 0;
    for (uint32_t i = orders_[h].prev; i != kInvalid; i = orders_[i].prev) sum += orders_[i].leaves;
    return sum;
  }

  // 依排隊順序走訪某價位的訂單。f(uint32_t handle, const OrderNode&) 回傳 false 可提前結束。
  template <class F>
  void forEachOrderAt(int64_t px, F&& f) const {
    const uint32_t lid = idx_.find(px);
    if (lid == kInvalid) return;
    for (uint32_t i = idx_.levelById(lid).head; i != kInvalid; i = orders_[i].next)
      if (!f(i, orders_[i])) return;
  }

  // 最佳價位隊首的訂單（例如預期最先被成交的單）。呼叫前需確認 !empty()。
  uint32_t bestFront() const { return idx_.levelById(idx_.bestLevel()).head; }

  const OrderNode& order(uint32_t h) const { return orders_[h]; }
  const Index& index() const { return idx_; }
  bool empty() const { return idx_.empty(); }
  int64_t bestPx() const { return idx_.bestPx(); }
  int64_t worstPx() const { return idx_.worstPx(); }
  uint32_t liveOrders() const { return live_; }
  void purgeEmpty() { idx_.purgeEmpty(); }

  // 測試用：檢查價位索引、FIFO 鏈結、數量彙總是否一致。
  bool checkInvariants() const {
    if (!idx_.checkInvariants()) return false;
    uint32_t seen = 0;
    bool ok = true;
    idx_.forEachBestToWorst([&](const PriceLevel& L) {
      int64_t sum = 0;
      uint32_t count = 0;
      uint32_t prev = kInvalid;
      for (uint32_t i = L.head; i != kInvalid; i = orders_[i].next) {
        const OrderNode& o = orders_[i];
        if (!o.live || o.prev != prev || o.px != L.px || &idx_.levelById(o.levelId) != &L || o.leaves <= 0 ||
            count > OrderCap) {
          ok = false;
          return false;
        }
        sum += o.leaves;
        ++count;
        prev = i;
      }
      if (prev != L.tail || sum != L.totalQty || count != L.orderCount) ok = false;
      seen += count;
      return ok;
    });
    return ok && seen == live_;
  }

 private:
  void linkBack(uint32_t h, uint32_t lid) {
    PriceLevel& L = idx_.levelById(lid);
    OrderNode& o = orders_[h];
    o.prev = L.tail;
    o.next = kInvalid;
    if (L.tail != kInvalid) orders_[L.tail].next = h;
    else L.head = h;
    L.tail = h;
  }

  void unlink(uint32_t h) {
    OrderNode& o = orders_[h];
    PriceLevel& L = idx_.levelById(o.levelId);
    if (o.prev != kInvalid) orders_[o.prev].next = o.next;
    else L.head = o.next;
    if (o.next != kInvalid) orders_[o.next].prev = o.prev;
    else L.tail = o.prev;
    o.prev = o.next = kInvalid;
  }

  Index idx_;
  OrderNode orders_[OrderCap];
  uint32_t freeHead_ = kInvalid;
  uint32_t live_ = 0;
};

}  // namespace oms
