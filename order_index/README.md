# order_index：策略自有掛單的價位索引

這個目錄是獨立的 header-only C++20 範例，和 xgbfast 函式庫無關，也不會被 xgbfast 的 build 或 CI 編到。

## 設計

- 價格一律用整數 tick 表示。
- 每一邊（買或賣）用一個**雙端排序陣列**，資料從中間開始放，兩端都留空間：
  - 新的最佳價（積極回補）放到尾端，O(1)
  - 新的最差價（遠端被動單）放到頭端，O(1)
  - 插在中間時，往比較近的一端 memmove
- 內部以 key 排序：買方 `key = px`，賣方 `key = -px`，所以最佳價永遠在尾端。
- `PriceLevel` 放在固定容量的 pool 裡。`levelId` 不會因為陣列搬移而改變，訂單可以直接存它。
- hot path 不配置記憶體、不丟例外。容量滿時 `addOrder` 回傳 `false`，由呼叫端決定要不要拒單。
- 查找時，價位數 ≤ 32 用線性掃描（從最佳價端開始），超過改用二分搜尋。
- **空價位延遲回收**（`SideIndex<Side, Cap, MaxEmpty>`）：
  - 價位的訂單數歸零時先不回收，因為刪單後常常馬上在同價位重下。
  - 不變式：**陣列兩端一定是有掛單的價位**。端點清空時會連同相鄰的空價位一起回收，所以 `bestPx()` 和 `worstPx()` 仍是 O(1)。
  - 空價位超過 `MaxEmpty` 個，或容量已滿又需要新價位時，做一次 O(n) 壓縮。呼叫端也可以在空閒時間呼叫 `purgeEmpty()`，把這個成本移出 hot path。
  - `MaxEmpty = 0` 等同立即回收。
- **每個價位的 FIFO 訂單佇列**（`OrderSide<Side, LevelCap, OrderCap, MaxEmpty>`）：
  - 訂單放在固定容量的池中，handle 在訂單存活期間不變，可以編進 clOrdId。
  - 每個價位的訂單用侵入式雙向鏈結串起來，`PriceLevel` 裡存 `head`/`tail`。
  - 排隊優先權：成交和減量保留原位置，加量排到隊尾，改價移到新價位的隊尾。
  - 改價時先建立新價位，再離開舊價位，所以失敗時狀態完全不變。例外是價位容量全滿、但舊價位只剩這張單：這時先離開舊價位騰出空間，一定成功。
  - `qtyAhead(h)` 回傳排在這張單前面的量，`bestFront()` 回傳最佳價隊首。

## 檔案

| 檔案 | 內容 |
|---|---|
| `side_index.hpp` | `LevelPool`、`SideIndex<Side, Cap, MaxEmpty>` |
| `order_side.hpp` | `OrderSide`：價位索引 + 訂單池 + 每個價位的 FIFO |
| `example.cpp` | 使用範例：一般掛單、積極回補、遠端被動單、排隊順序 |
| `fuzz_side_index.cpp` | `SideIndex` 的 fuzz 測試（以 `std::map` 為參考）加確定性測試（含延遲回收） |
| `fuzz_order_side.cpp` | `OrderSide` 的 fuzz 測試（以 `std::map<px, vector<訂單>>` 為參考）加確定性測試 |
| `bench_side_index.cpp` | 粗略微基準，跟 `std::map` 做相對比較 |

## 建置與執行

```bash
cd order_index
make test        # 兩組 fuzz
make test-san    # AddressSanitizer + UBSan 版本
make bench
./fuzz_order_side 100000 200 1   # 自訂：每個 seed 的步數、seed 數、起始 seed
```

## fuzz 涵蓋的內容

- 事件分布：90% 的價位落在 mid ± 20 tick，5% 是積極價，5% 是遠端被動價。極端價位傾向很快被移除，mid 偶爾會跳空。
- 容量分成 8、48、256 三種：8 會頻繁撞到兩端和容量上限，48 橫跨線性掃描和二分搜尋的切換點。
- 每一步之後檢查：內部不變式（嚴格排序、pool 使用量等於價位數、`head ≤ tail`）、最佳價、最差價、最佳到最差的完整走訪（價格、量、單數），以及存在和不存在價位的精確查找。
- `SideIndex` 另外用 `MaxEmpty` = 0、3、8、32 測試延遲回收，並隨機插入空閒時的 `purgeEmpty()`。

`OrderSide` 的 fuzz 事件包含：新單、最佳價隊首成交、隨機部分成交、刪單、同價改量（加量或減量）、改價（含改到極端價位），以及 mid 跳空。組態刻意讓價位容量和訂單池分別成為瓶頸。每一步之後比對：

- 內部不變式：鏈結的前後指標一致、每個價位的量和單數彙總等於鏈結內容
- 價位彙總
- **每個價位的 FIFO 順序**（handle、clOrdId、剩餘量）
- `qtyAhead`
- `bestFront`
- 容量不足時的拒絕條件

我也刻意在程式裡注入過幾種錯誤來確認測試抓得到，例如加量不排到隊尾、端點空價位沒回收、重新啟用空價位時計數沒減、unlink 沒更新 tail、`find` 回傳空價位等，每一種都會被測試抓到。
