# xgbfast

Compile XGBoost once; predict from Python and C++ through a small native interface.

Version 0.3.0 is under development and is not published on PyPI. Linux ARM64 has
been tested; x86-64 CI is configured but has not been run in the local session.
TL2cgen generates the model's C implementation. xgbfast adds compilation,
prediction validation, artifact metadata, and Python/C++ runtime interfaces.

## Install on Linux

Python 3.12 is the tested development version. From this repository's root:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential cmake git ca-certificates libgomp1 python3-dev python3-venv
python3 -m venv .venv
source .venv/bin/activate
CXXFLAGS="${CXXFLAGS:-} -include cstdint" python -m pip install '.[compiler]'
```

The CXXFLAGS setting fixes a missing header in the Treelite source fetched by
TL2cgen 1.0.0 when building from source on ARM64 with GCC 13.
Runtime-only source installation is `python -m pip install .`; it still needs
a C++20 compiler and Python headers. Installing a compatible prebuilt wheel
does not need a compiler. Wheels are specific to Python, OS, and architecture.

## Compile a model

Save an existing XGBoost model with `trained_model.save_model("model.json")`, then:

```python
from xgbfast import compile_model

compile_model("model.json", "compiled/my_model", validation_data="X_test.npy")
```

Or use the CLI:

```bash
xgbfast compile model.json --output compiled/my_model --validate-data X_test.npy
xgbfast inspect compiled/my_model
xgbfast validate compiled/my_model --data X_test.npy
```

Representative validation data is an optional nonempty 2D NumPy `.npy` file.
Builds always perform synthetic parity checks before publishing the artifact.
Use a new output directory when the model or build configuration changes.
Keep the entire artifact directory for Python deployment; compile for the
deployment OS, CPU architecture, and compatible libc.

## Python

```python
import numpy as np
from xgbfast import Predictor

with Predictor.load("compiled/my_model", mode="native") as model:
    features = np.zeros(model.num_features, dtype=np.float32)
    # Update features in training-time order for each prediction.
    score = model.predict(features)
```

Native mode accepts an aligned, contiguous, one-dimensional native float32
buffer, including NumPy arrays and `array.array('f')`. It returns an owned
Python float. NaN means missing; infinity is rejected. Allocate the model and
input once and reuse them. Create, use, and close each model on its owning worker.
The short native call holds the GIL; use processes for parallel Python inference.

| Mode | Input / usage |
|---|---|
| `native` | Low-latency single float32 vector; CPython extension |
| `direct` | Strict float32 vector or batch; ctypes; one instance per worker |
| `threaded` (default) | Input conversion, batches, shared instances with per-thread contexts |

## C++20

```cpp
#include <xgbfast.hpp>
#include <vector>

xgbfast::Model model("compiled/my_model/model.so");
std::vector<float> features(model.num_features(), 0.0f);
float score = model.predict(features);
```

```bash
c++ -O3 -std=c++20 -I src/xgbfast/include examples/predict.cpp -ldl -o /tmp/xgbfast-predict
/tmp/xgbfast-predict compiled/my_model/model.so
```

The header-only wrapper loads the model library and owns its resources. Each
worker owns a separate model. C++ does not need Python or ML runtime packages.
Installed header paths are available from `xgbfast.get_include()`.
The C++ loader checks ABI and feature count, but does not parse the manifest or
verify hashes. Python's factory performs those additional artifact checks.
Load model libraries only from trusted builds.

## Test and benchmark

```bash
python scripts/install_bench.py
python -m pip install .
python -m unittest discover -s tests -v
# Creates a fixed-seed demo, validates it, and benchmarks all Python backends:
python scripts/validate_linux.py --output results/linux
```

The Linux validator needs no external model or files from another repository.
Optionally pass `--model model.json --data X_test.npy --cpu 0` (choose an allowed CPU).
It checks standalone model dependencies and runtime imports, and preserves raw
latency samples plus metadata. CI runs the same flow on Linux x86-64 and ARM64.

```bash
docker build -t xgbfast-linux .
docker run --rm -v "$PWD/results:/results" xgbfast-linux
```

Earlier Linux ARM64 VM measurements for a 100-tree, depth-4, 32-feature model
gave single-row p50 of 1.416 µs for the ctypes prototype and 0.375 µs for native
Python. These are warm VM measurements, not a production latency guarantee.
The standalone benchmark produces its own results; C++ latency is not measured
by the Python comparison.

## Current scope

- Numeric XGBoost `gbtree`, single-target `binary:logistic` or `reg:squarederror`.
- JSON/UBJ saved models; categorical and multi-target models are rejected.
- Single-row native Python/C++; batch support through the existing ctypes API.
- Treelite/TL2cgen remain build dependencies. Their versions and licenses remain separate.
- A project license has not yet been selected; no open-source license is granted by this repository.

[Detailed usage (繁體中文)](LIBRARY.md) · [Development](CONTRIBUTING.md) · [GitHub preparation](GITHUB.md)
