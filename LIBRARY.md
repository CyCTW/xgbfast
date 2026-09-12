# Python 與 C++ library

0.3.0 提供共用的 C++20 RAII 封裝及 CPython 原生擴充。模型仍由 TL2cgen 生成 C，
產物使用既有 ABI 1，因此可以載入先前編譯的同平台模型。

## 編譯一次

```python
from xgbfast import compile_model
compile_model("model.json", "compiled/my_model", validation_data="X_test.npy")
```

模型編譯需要 XGBoost／Treelite／TL2cgen 與 C compiler。推論不需要這些 Python 套件。
從原始碼安裝 0.3.0 需要 C++20 compiler 與對應 Python 開發標頭；Ubuntu 安裝 `build-essential python3-dev`。
wheel 現在依 Python 版本、OS 與 CPU 架構建置，不能沿用 0.2 的通用 wheel。

## Python 單筆低延遲 API

```python
import numpy as np
from xgbfast import Predictor

with Predictor.load("compiled/my_model", mode="native") as model:
    features = np.zeros(model.num_features, dtype=np.float32)
    score = model.predict(features)
```

接受一維、對齊、連續的 native float32 buffer，包括 NumPy array、`array.array('f')` 和相容 memoryview。
每次呼叫直接回傳可保留的 Python float；不經 ctypes、DMatrix、NumPy scalar 轉換，也不配置輸入或 scratch。
Python float 與 buffer 協定仍有呼叫成本。
目前 native API 專注單筆，不接受二維批次或 list 自動轉型；批次使用既有 `mode="direct"` 或 `mode="threaded"`。

每個 worker 建立自己的模型，並在同一 worker 呼叫與關閉。短時間單筆推論持有 GIL，
避免释放／重取 GIL 的成本；因此 Python threads 不會藉此平行推論，平行需求使用獨立 process。
不要由外部 native writer 同時修改輸入。NaN 表示 missing，infinity 拒絕。

## C++ 單筆 API

```cpp
#include <xgbfast.hpp>
#include <vector>

xgbfast::Model model("compiled/my_model/model.so");
std::vector<float> features(model.num_features(), 0.0f);
float score = model.predict(features);
```

`predict` 接受 `std::span<const float>`，可直接使用 vector、array 或既有 float buffer。
建構時載入 library 與配置 scratch；推論成功路徑不配置 heap，析構自動釋放。
模型禁止複製，一個 worker 擁有一個實例；close 可重複，錯誤以例外回報。

從 repository 根目錄編譯範例：

```bash
c++ -O3 -std=c++20 -I src/xgbfast/include \
  examples/predict.cpp -ldl -o predict
./predict compiled/my_model/model.so
```

header 也隨 Python wheel 安裝，`xgbfast.get_include()` 回傳其位置。
C++ 部署只需要 header、編譯好的模型與系統 C/C++ libraries，不需要 Python。
目前 C++ 直接載入可信任的 `.so`／`.dylib` 並驗證 ABI／feature count；不解析 JSON manifest 或驗證檔案 hash。
Python factory 則會先驗證 manifest、平台與 hash。動態 library 必須來自可信任的建置流程。

## C ABI

`xgbfast.h` 公開 ABI 1：`xf_create`／`xf_predict`／`xf_free`，可供其他語言接入。
呼叫者提供 row-major float32 輸入、獨立輸出與容量，每個並行 caller 使用獨立 context。
錯誤時輸出可能部分寫入；檢查 status 後才使用結果。
