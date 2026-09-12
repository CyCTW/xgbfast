"""NumPy + ctypes only. Native resources are isolated per calling thread."""
import ctypes as ct
import threading
from pathlib import Path

import numpy as np

from .artifact import ABI_VERSION, CompatibilityError, inspect_model

FLOAT32 = np.dtype("float32")


class _Session:
    def __init__(self, library):
        self.library = library  # Keep function pointers/library alive through cleanup.
        self.lock = threading.Lock()
        self.handle = library.xf_create()
        if not self.handle:
            raise RuntimeError("Unable to allocate native predictor context")
        self.scalar = np.empty(1, dtype=np.float32)
        self.scalar_address = self.scalar.ctypes.data

    def close(self):
        with self.lock:
            if self.handle:
                self.library.xf_free(self.handle)
                self.handle = None

    def __del__(self):
        if getattr(self, "handle", None):
            self.close()


class Predictor:
    """Load via Predictor.load(). 1D input returns float; 2D input returns (N,).

    predict_into requires aligned, C-contiguous float32 NumPy arrays and a
    writable output of shape (N,). Callers must not mutate inputs or share output
    buffers concurrently. Calls on different threads use separate C contexts.
    """

    @classmethod
    def load(cls, artifact, *, mode="threaded"):
        if mode == "native":
            from ._native import Model
            manifest = inspect_model(artifact)
            model = Model(str(Path(artifact).resolve() / manifest["library"]["file"]))
            if model.num_features != manifest["num_features"]:
                model.close()
                raise CompatibilityError("Native feature count differs from manifest")
            return model
        if mode == "direct":
            return DirectPredictor(artifact)
        if mode != "threaded":
            raise ValueError("mode must be 'native', 'direct' or 'threaded'")
        return cls(artifact)

    def __init__(self, artifact):
        self._closed = False
        self._registry_lock = threading.Lock()
        self._local = threading.local()
        self._sessions = []
        self.manifest = inspect_model(artifact)
        self.num_features = self.manifest["num_features"]
        self.feature_names = self.manifest.get("feature_names")
        self.output_kind = self.manifest["output_kind"]
        self._lib = ct.CDLL(str(Path(artifact).resolve() / self.manifest["library"]["file"]))
        lib = self._lib
        lib.xf_abi_version.argtypes, lib.xf_abi_version.restype = [], ct.c_uint32
        lib.xf_num_features.argtypes, lib.xf_num_features.restype = [], ct.c_int32
        if lib.xf_abi_version() != ABI_VERSION or lib.xf_num_features() != self.num_features:
            raise CompatibilityError("Native library ABI or feature count differs from manifest")
        lib.xf_create.argtypes, lib.xf_create.restype = [], ct.c_void_p
        lib.xf_free.argtypes, lib.xf_free.restype = [ct.c_void_p], None
        lib.xf_predict.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t,
                                  ct.c_size_t, ct.c_void_p, ct.c_size_t]
        lib.xf_predict.restype = ct.c_int
        self._native = lib.xf_predict
        self._session()  # Allocate this thread's scalar/scratch storage at load time.

    def _session(self):
        session = getattr(self._local, "session", None)
        if session is None:
            with self._registry_lock:
                if self._closed:
                    raise RuntimeError("Predictor is closed")
                session = _Session(self._lib)
                self._sessions.append(session)
                self._local.session = session
        return session

    def _input(self, value, strict):
        if strict:
            if (not isinstance(value, np.ndarray) or value.dtype != FLOAT32
                    or not value.flags.c_contiguous or not value.flags.aligned):
                raise ValueError("predict_into input must be aligned C-contiguous float32 NumPy data")
            data = value
        else:
            data = np.asarray(value, dtype=np.float32)
            if not data.flags.c_contiguous or not data.flags.aligned:
                data = np.require(data, dtype=np.float32, requirements=["C", "A"])
        if data.ndim not in (1, 2) or data.shape[-1] != self.num_features:
            raise ValueError(f"Expected ({self.num_features},) or (N, {self.num_features}); got {data.shape}")
        return data, 1 if data.ndim == 1 else data.shape[0]

    def _run(self, session, data, rows, address):
        if not session.handle:
            raise RuntimeError("Predictor is closed")
        status = self._native(session.handle, data.ctypes.data, rows, self.num_features, address, rows)
        if status == 2:
            raise ValueError("Input contains infinity; use NaN for missing features")
        if status:
            raise RuntimeError(f"Native prediction failed with status {status}")

    def predict(self, features):
        data, rows = self._input(features, strict=False)
        session = self._session()
        if data.ndim == 1:
            with session.lock:
                self._run(session, data, 1, session.scalar_address)
                return float(session.scalar[0])
        output = np.empty(rows, dtype=np.float32)
        with session.lock:
            self._run(session, data, rows, output.ctypes.data)
        return output

    def predict_into(self, features, out):
        data, rows = self._input(features, strict=True)
        if (not isinstance(out, np.ndarray) or out.dtype != FLOAT32 or out.shape != (rows,)
                or not out.flags.c_contiguous or not out.flags.aligned or not out.flags.writeable):
            raise ValueError(f"Output must be writable, aligned, C-contiguous float32 with shape ({rows},)")
        if np.may_share_memory(data, out):
            raise ValueError("Input and output buffers must not overlap")
        session = self._session()
        with session.lock:
            self._run(session, data, rows, out.ctypes.data)

    def close(self):
        # Wait for any in-flight call before releasing that thread's context.
        with self._registry_lock:
            self._closed = True
            for session in self._sessions:
                session.close()
            self._sessions.clear()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Predictor is closed")
        return self

    def __exit__(self, *unused):
        self.close()

    def __del__(self):
        if hasattr(self, "_sessions"):
            self.close()


class DirectPredictor:
    """Thin, thread-confined wrapper around the same generated C predict().

    Construct one instance per worker. No TLS lookup or lock on prediction.
    Inputs must already be aligned, C-contiguous float32 arrays. No DMatrix,
    Python feature loop, or input conversion is used. Scalar output is reused
    internally and returned as an owned float. Reentrant calls are rejected.
    """

    def __init__(self, artifact):
        self._base = Predictor(artifact)  # Schema/ABI checks and allocation once.
        self._owner = threading.get_ident()
        self._busy = False
        self.manifest = self._base.manifest
        self.num_features = self._base.num_features
        self.feature_names = self._base.feature_names
        self.output_kind = self._base.output_kind
        session = self._base._session()
        self._handle = session.handle
        self._scalar = session.scalar
        self._scalar_address = session.scalar_address
        self._native = self._base._native

    def predict(self, features):
        if threading.get_ident() != self._owner:
            raise RuntimeError("Direct predictor is thread-confined; create one per worker")
        if not self._handle or self._busy:
            raise RuntimeError("Direct predictor is closed or already in use")
        if (not isinstance(features, np.ndarray) or features.dtype != FLOAT32
                or features.ndim not in (1, 2) or features.shape[-1] != self.num_features
                or not features.flags.c_contiguous or not features.flags.aligned):
            raise ValueError(f"Direct input must be aligned C-contiguous float32 with shape ({self.num_features},) or (N, {self.num_features})")
        self._busy = True
        try:
            if features.ndim == 1:
                status = self._native(self._handle, features.ctypes.data, 1,
                                      self.num_features, self._scalar_address, 1)
                if status:
                    self._raise_status(status)
                return float(self._scalar[0])
            rows = features.shape[0]
            out = np.empty(rows, dtype=np.float32)
            status = self._native(self._handle, features.ctypes.data, rows,
                                  self.num_features, out.ctypes.data, rows)
            if status:
                self._raise_status(status)
            return out
        finally:
            self._busy = False

    def predict_into(self, features, out):
        if threading.get_ident() != self._owner:
            raise RuntimeError("Direct predictor is thread-confined; create one per worker")
        if not self._handle or self._busy:
            raise RuntimeError("Direct predictor is closed or already in use")
        data, rows = self._base._input(features, strict=True)
        if (not isinstance(out, np.ndarray) or out.dtype != FLOAT32 or out.shape != (rows,)
                or not out.flags.c_contiguous or not out.flags.aligned or not out.flags.writeable):
            raise ValueError(f"Output must be writable aligned C-contiguous float32 with shape ({rows},)")
        if np.may_share_memory(data, out):
            raise ValueError("Input and output buffers must not overlap")
        self._busy = True
        try:
            status = self._native(self._handle, data.ctypes.data, rows,
                                  self.num_features, out.ctypes.data, rows)
            if status:
                self._raise_status(status)
        finally:
            self._busy = False

    @staticmethod
    def _raise_status(status):
        if status == 2:
            raise ValueError("Input contains infinity; use NaN for missing features")
        raise RuntimeError(f"Native prediction failed with status {status}")

    def close(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError("Close a direct predictor on its owning worker")
        if self._busy:
            raise RuntimeError("Cannot close a direct predictor during a prediction")
        self._base.close()
        self._handle = None

    def __enter__(self):
        if not self._handle:
            raise RuntimeError("Direct predictor is closed")
        return self

    def __exit__(self, *unused):
        self.close()

    def __del__(self):
        # At final collection no method can still be holding this object alive.
        # The last reference can be dropped on a different thread.
        if hasattr(self, "_base"):
            self._base.close()
