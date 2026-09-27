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

## 檔案

| 檔案 | 內容 |
|---|---|
| `side_index.hpp` | `LevelPool`、`SideIndex<Side, Cap>` |
| `example.cpp` | 使用範例：一般掛單、積極回補、遠端被動單 |
| `fuzz_side_index.cpp` | 以 `std::map` 為參考實作的 fuzz 測試，另含幾個針對邊界的確定性測試 |
| `bench_side_index.cpp` | 粗略微基準，跟 `std::map` 做相對比較 |

## 建置與執行

```bash
cd order_index
make test        # fuzz：預設 50 個 seed × 20000 步 × 6 種組態
make test-san    # AddressSanitizer + UBSan 版本
make bench
./fuzz_side_index 100000 200 1   # 自訂：每個 seed 的步數、seed 數、起始 seed
```

## fuzz 涵蓋的內容

- 事件分布：90% 的價位落在 mid ± 20 tick，5% 是積極價，5% 是遠端被動價。極端價位傾向很快被移除，mid 偶爾會跳空。
- 容量分成 8、48、256 三種：8 會頻繁撞到兩端和容量上限，48 橫跨線性掃描和二分搜尋的切換點。
- 每一步之後檢查：內部不變式（嚴格排序、pool 使用量等於價位數、`head ≤ tail`）、最佳價、最差價、最佳到最差的完整走訪（價格、量、單數），以及存在和不存在價位的精確查找。
