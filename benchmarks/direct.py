"""Python -> generated C prediction, with reusable scratch and output buffers."""
import ctypes
import json
from pathlib import Path

import numpy as np
import tl2cgen


def compile_predictor(model, source_dir, toolchain):
    """One library serves both the normal TL2cgen runtime and the direct path."""
    source_dir = Path(source_dir)
    tl2cgen.generate_c_code(model, dirpath=source_dir, params={"parallel_comp": 4})
    header = (source_dir / "header.h").read_text()
    # Reject incompatible generator versions/types instead of guessing their ABI.
    required = ("void predict(union Entry* data, int pred_margin, float* result);",
                "float fvalue;", "#define N_TARGET 1", "#define MAX_N_CLASS 1")
    if not all(declaration in header for declaration in required):
        raise ValueError("Direct path requires TL2cgen's scalar float32 predict ABI")
    source = Path(__file__).with_name("direct.c").read_text()
    (source_dir / "direct.c").write_text(source)
    recipe_path = source_dir / "recipe.json"
    recipe = json.loads(recipe_path.read_text())
    recipe["sources"].append({"name": "direct", "length": len(source.splitlines())})
    recipe_path.write_text(json.dumps(recipe, indent=2))
    return Path(tl2cgen.create_shared(toolchain, source_dir, nthread=4))


class DirectPredictor:
    """Serial caller only. Returned arrays alias buffers overwritten on reuse.

    Call prepare(nrow) before timing to allocate output for that batch size.
    Input address lookup, checks, ctypes call, C feature copy, NaN conversion,
    output reset, and generated model inference all happen in __call__.
    """

    def __init__(self, lib_path):
        self._lib = ctypes.CDLL(str(Path(lib_path).resolve()))
        self._lib.direct_create.argtypes = []
        self._lib.direct_create.restype = ctypes.c_void_p
        self._lib.direct_free.argtypes = [ctypes.c_void_p]
        self._lib.direct_free.restype = None
        self._lib.get_num_feature.argtypes = []
        self._lib.get_num_feature.restype = ctypes.c_int32
        self.num_feature = self._lib.get_num_feature()
        self._predict = self._lib.direct_predict
        self._predict.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.c_size_t, ctypes.c_void_p]
        self._predict.restype = None
        self._handle = self._lib.direct_create()
        if not self._handle:
            raise RuntimeError("Unable to allocate direct predictor or unsupported model types")
        self._outputs = {}

    def prepare(self, nrow):
        if nrow < 1:
            raise ValueError("Expected at least one row")
        if nrow not in self._outputs:
            output = np.empty(nrow, dtype=np.float32)
            self._outputs[nrow] = (output, output.ctypes.data)

    def __call__(self, data):
        if not self._handle:
            raise RuntimeError("Direct predictor is closed")
        if (not isinstance(data, np.ndarray) or data.dtype != np.float32
                or data.ndim != 2 or data.shape[1] != self.num_feature
                or not data.flags.c_contiguous):
            raise ValueError(f"Expected C-contiguous float32 input of shape (N, {self.num_feature})")
        nrow = data.shape[0]
        # An unprepared batch size is an error, not a hidden timed allocation.
        output, address = self._outputs[nrow]
        self._predict(self._handle, data.ctypes.data, nrow, address)
        return output

    def close(self):
        if getattr(self, "_handle", None):
            self._lib.direct_free(self._handle)
            self._handle = None

    def __del__(self):
        self.close()
