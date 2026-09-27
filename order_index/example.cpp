// SideIndex 使用範例：一般掛單 + 積極回補 + 遠端被動單。
#include "order_side.hpp"

#include <cstdio>

using namespace oms;

template <class Index>
void dump(const char* title, const Index& idx) {
  std::printf("%s (levels=%u)\n", title, idx.size());
  idx.forEachBestToWorst([](const PriceLevel& L) {
    std::printf("  px=%-10lld qty=%-6lld orders=%u\n", (long long)L.px, (long long)L.totalQty, L.orderCount);
    return true;
  });
}

int main() {
  SideIndex<Side::Buy> bids;

  // 市價附近的一般掛單（價格單位：tick）
  bids.addOrder(10000, 5);
  bids.addOrder(9999, 3);
  bids.addOrder(9998, 7);
  bids.addOrder(9999, 2);  // 同價位第二張單
  dump("初始買方", bids);

  // 回補：積極買單掛很高 → O(1) 放在尾端，立刻成為最佳價
  bids.addOrder(12000, 10);
  // 遠端被動單 → O(1) 放在頭端
  bids.addOrder(5000, 1);
  dump("加入積極價與遠端價", bids);
  std::printf("best=%lld worst=%lld\n", (long long)bids.bestPx(), (long long)bids.worstPx());

  // 查詢某價位
  if (const PriceLevel* L = bids.level(9999))
    std::printf("9999 價位: qty=%lld orders=%u\n", (long long)L->totalQty, L->orderCount);

  // 積極單部分成交 4 口，之後剩餘 6 口刪單
  bids.reduce(12000, 4, /*removeOrder=*/false);
  bids.reduce(12000, 6, /*removeOrder=*/true);  // 價位清空後自動移除
  dump("積極單成交並刪除後", bids);
  std::printf("best=%lld\n", (long long)bids.bestPx());

  // ---- OrderSide：價位 + 每個價位的 FIFO 訂單佇列 ----
  OrderSide<Side::Sell> asks;
  const uint32_t a1 = asks.add(/*clOrdId=*/101, 10005, 10);
  const uint32_t a2 = asks.add(102, 10005, 20);
  const uint32_t a3 = asks.add(103, 10006, 5);
  std::printf("\n賣方最佳價=%lld 隊首=%llu，a2 前方量=%lld\n", (long long)asks.bestPx(),
              (unsigned long long)asks.order(asks.bestFront()).clOrdId, (long long)asks.qtyAhead(a2));

  asks.reduce(a1, 4);       // 部分成交：保留排隊位置
  asks.modifyQty(a1, 30);   // 加量：排到隊尾
  std::printf("a1 加量後 10005 的排隊順序:");
  asks.forEachOrderAt(10005, [](uint32_t, const OrderNode& o) {
    std::printf(" #%llu(%lld)", (unsigned long long)o.clOrdId, (long long)o.leaves);
    return true;
  });
  std::printf("\n");

  // 回補：a3 改價到很低的積極賣價，handle 不變
  asks.replace(a3, 9000, 5);
  std::printf("a3 改價後 best=%lld 隊首=%llu\n", (long long)asks.bestPx(),
              (unsigned long long)asks.order(asks.bestFront()).clOrdId);
  asks.remove(a3);  // 成交完畢或刪單
  std::printf("a3 移除後 best=%lld，空價位保留數=%u\n", (long long)asks.bestPx(), asks.index().emptyCount());
  return 0;
}
