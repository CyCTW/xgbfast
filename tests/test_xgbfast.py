import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import xgboost as xgb

from xgbfast import CompatibilityError, Predictor, compile_model, inspect_model, validate_model


class PackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        rng = np.random.default_rng(17)
        cls.data = rng.normal(size=(100, 4)).astype(np.float32)
        cls.data[::7, 1] = np.nan
        labels = cls.data[:, 0] * 2 - np.nan_to_num(cls.data[:, 1])
        cls.cases = []
        for index, objective in enumerate(("binary:logistic", "reg:squarederror")):
            y = labels > 0 if index == 0 else labels
            booster = xgb.train({"objective": objective, "max_depth": 3, "nthread": 1},
                                xgb.DMatrix(cls.data, label=y, feature_names=["a", "b", "c", "d"]),
                                num_boost_round=9)
            if index == 1:
                booster.set_attr(best_iteration="4")
            source = cls.root / ("classifier.json" if index == 0 else "regressor.ubj")
            booster.save_model(source)
            artifact = compile_model(source, cls.root / f"artifact{index}", validation_data=cls.data, jobs=1)
            reference = booster if index == 0 else booster[:5]
            cls.cases.append((source, artifact, reference))

    def test_native_vector_parity_and_contract(self):
        from array import array
        for _, artifact, reference in self.cases:
            with Predictor.load(artifact, mode="native") as model:
                expected = reference.inplace_predict(self.data)
                actual = [model.predict(row) for row in self.data]
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
                self.assertIsInstance(actual[0], float)
                self.assertEqual(model.num_features, 4)
                self.assertAlmostEqual(model.predict(array("f", self.data[0])), actual[0])
                invalid = [self.data, self.data[0].astype(np.float64), self.data[0][::2],
                           np.zeros(3, dtype=np.float32), np.full(4, np.inf, dtype=np.float32),
                           np.zeros(4, dtype=">f4"),
                           np.ndarray((4,), dtype=np.float32, buffer=bytearray(17), offset=1)]
                for value in invalid:
                    with self.assertRaises(ValueError):
                        model.predict(value)
                with ThreadPoolExecutor(max_workers=1) as pool:
                    with self.assertRaises(RuntimeError):
                        pool.submit(model.predict, self.data[0]).result()
                    with self.assertRaises(RuntimeError):
                        pool.submit(model.close).result()
                model.close()
                model.close()
                with self.assertRaises(RuntimeError):
                    model.predict(self.data[0])

    def test_cpp_library_parity_and_lifecycle(self):
        import xgbfast
        compiler = shutil.which("c++")
        self.assertIsNotNone(compiler)
        source = self.root / "consumer.cpp"
        source.write_text(r'''
#include "xgbfast.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
int main(int argc, char** argv) {
    xgbfast::Model model(argv[1]);
    std::ifstream input(argv[2], std::ios::binary);
    std::vector<float> row(model.num_features());
    while (input.read(reinterpret_cast<char*>(row.data()), row.size()*sizeof(float)))
        std::cout << std::setprecision(9) << model.predict(row) << '\n';
    try { model.predict(std::span<const float>{}); return 2; }
    catch (const std::invalid_argument&) {}
    model.close(); model.close();
    try { model.predict(row); return 3; }
    catch (const std::runtime_error&) {}
}
''')
        executable = self.root / "consumer"
        command = [compiler, "-std=c++20", "-O3", "-I",
                   str(Path(xgbfast.__file__).parent / "include"), str(source), "-o", str(executable)]
        if sys.platform == "linux":
            command.append("-ldl")
        subprocess.run(command, check=True, capture_output=True)
        data = self.root / "vectors.bin"
        self.data.tofile(data)
        for _, artifact, reference in self.cases:
            manifest = inspect_model(artifact)
            output = subprocess.run([str(executable), str(artifact / manifest["library"]["file"]), str(data)],
                                    check=True, capture_output=True, text=True).stdout
            np.testing.assert_allclose(np.fromstring(output, sep="\n"), reference.inplace_predict(self.data),
                                       rtol=1e-5, atol=1e-6)

    def test_native_repeated_load_releases_type_references(self):
        from xgbfast._native import Model
        _, artifact, _ = self.cases[0]
        library = artifact / inspect_model(artifact)["library"]["file"]
        before = sys.getrefcount(Model)
        for _ in range(50):
            model = Model(str(library))
            model.predict(self.data[0])
            del model
        self.assertEqual(sys.getrefcount(Model), before)

    def test_scalar_batch_conversion_and_ownership(self):
        for _, artifact, reference in self.cases:
            with self.subTest(artifact=str(artifact)), Predictor.load(artifact) as model:
                expected = reference.inplace_predict(self.data)
                self.assertEqual(model.feature_names, ["a", "b", "c", "d"])
                value = model.predict(self.data[0])
                self.assertIsInstance(value, float)
                self.assertAlmostEqual(value, float(expected[0]), places=6)
                output = model.predict(self.data)
                snapshot = output.copy()
                np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-6)
                for data in (self.data.tolist(), self.data.astype(np.float64)):
                    np.testing.assert_allclose(model.predict(data), expected, rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(model.predict(self.data[::-1]), expected[::-1], rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(model.predict(self.data[::2]), expected[::2], rtol=1e-5, atol=1e-6)
                model.predict(np.zeros_like(self.data))
                np.testing.assert_array_equal(output, snapshot)
                self.assertAlmostEqual(value, float(expected[0]), places=6)
                self.assertEqual(model.predict(np.empty((0, 4), dtype=np.float32)).shape, (0,))

    def test_predict_into_repeated_missing_and_output_reset(self):
        batches = [self.data[:7], np.full((7, 4), np.nan, dtype=np.float32),
                   np.zeros((7, 4), dtype=np.float32), self.data[10:17]]
        for _, artifact, reference in self.cases:
            with Predictor.load(artifact) as model:
                out = np.empty(7, dtype=np.float32)
                for data in batches * 8:
                    before = data.copy()
                    self.assertIsNone(model.predict_into(data, out))
                    np.testing.assert_allclose(out, reference.inplace_predict(data), rtol=1e-5, atol=1e-6)
                    np.testing.assert_array_equal(data, before)

    def test_invalid_input_output_and_native_bounds(self):
        with Predictor.load(self.cases[0][1]) as model:
            for bad in ([], 2.0, [1, 2], np.zeros((1, 1, 4))):
                with self.subTest(bad=str(bad)), self.assertRaises(ValueError):
                    model.predict(bad)
            for bad in ([1, 2, 3, 4], np.zeros(4, dtype=np.float64), self.data[::2]):
                with self.assertRaises(ValueError):
                    model.predict_into(bad, np.empty(1, dtype=np.float32))
            readonly = np.empty(1, dtype=np.float32)
            readonly.flags.writeable = False
            for out in (readonly, np.empty(1, dtype=np.float64), np.empty(2, dtype=np.float32)):
                with self.assertRaises(ValueError):
                    model.predict_into(self.data[0], out)
            with self.assertRaises(ValueError):
                model.predict_into(self.data[0], self.data[0, :1])
            unaligned = np.ndarray((4,), dtype=np.float32, buffer=bytearray(17), offset=1)
            unaligned[:] = 0
            self.assertIsInstance(model.predict(unaligned), float)
            with self.assertRaises(ValueError):
                model.predict_into(unaligned, np.empty(1, dtype=np.float32))
            with self.assertRaises(ValueError):
                model.predict([np.inf, 0, 0, 0])
            # C ABI independently rejects invalid capacity/features before touching pointers.
            native = model._native
            session = model._session()
            self.assertEqual(native(session.handle, None, 1, 4, None, 0), 1)
            self.assertEqual(native(session.handle, None, 1, 3, None, 1), 1)

    def test_threads_do_not_share_context_or_scalar(self):
        _, artifact, reference = self.cases[0]
        expected = reference.inplace_predict(self.data)
        with Predictor.load(artifact) as model:
            barrier = threading.Barrier(4)

            def worker(offset):
                barrier.wait(timeout=5)
                for index in range(offset, 100, 4):
                    for _ in range(10):
                        self.assertAlmostEqual(model.predict(self.data[index]), float(expected[index]), places=6)
                return model._session().handle

            with ThreadPoolExecutor(max_workers=4) as executor:
                handles = list(executor.map(worker, range(4)))
            self.assertEqual(len(set(handles)), 4)

    def test_close_waits_for_active_call_and_is_idempotent(self):
        model = Predictor.load(self.cases[0][1])
        self.addCleanup(model.close)
        entered, release, closed = threading.Event(), threading.Event(), threading.Event()
        original = model._native

        def delayed(*args):
            entered.set()
            self.assertTrue(release.wait(timeout=5))
            return original(*args)

        model._native = delayed
        with ThreadPoolExecutor(max_workers=2) as executor:
            prediction = executor.submit(model.predict, self.data[0])
            self.assertTrue(entered.wait(timeout=5))
            close = executor.submit(lambda: (model.close(), closed.set()))
            self.assertFalse(closed.wait(timeout=0.05))
            release.set()
            self.assertIsInstance(prediction.result(timeout=5), float)
            close.result(timeout=5)
        model.close()
        with self.assertRaises(RuntimeError):
            model.predict(self.data[0])

    def test_cache_and_early_stopping(self):
        source, artifact, _ = self.cases[1]
        before = (artifact / "manifest.json").read_bytes()
        with patch("tl2cgen.create_shared", side_effect=AssertionError("cache recompiled")):
            self.assertEqual(compile_model(source, artifact, jobs=1), artifact)
        self.assertEqual((artifact / "manifest.json").read_bytes(), before)
        self.assertEqual(inspect_model(artifact)["effective_rounds"], 5)
        with self.assertRaises(FileExistsError):
            compile_model(self.cases[0][0], artifact, jobs=1)

    def test_corrupt_or_incompatible_artifact_rejected_before_loading(self):
        for field in ("abi", "platform", "hash", "feature_count"):
            copy = self.root / f"corrupt-{field}"
            shutil.copytree(self.cases[0][1], copy)
            path = copy / "manifest.json"
            manifest = json.loads(path.read_text())
            if field == "abi":
                manifest["abi_version"] = 99
            elif field == "platform":
                manifest["target"]["machine"] = "different-architecture"
            elif field == "hash":
                manifest["library"]["sha256"] = "broken"
            else:
                manifest["num_features"] = 3
            path.write_text(json.dumps(manifest))
            with self.subTest(field=field), self.assertRaises(CompatibilityError):
                Predictor.load(copy)

    def test_failed_build_not_published_and_logs_saved(self):
        target = self.root / "failed"
        with patch("tl2cgen.create_shared", side_effect=RuntimeError("compiler failure")):
            with self.assertRaisesRegex(RuntimeError, "compiler failure"):
                compile_model(self.cases[0][0], target, jobs=1)
        self.assertFalse(target.exists())
        self.assertTrue((self.root / "failed.build-error.log").exists())
        self.assertEqual(list(self.root.glob(".failed.build-*")), [])
        self.assertFalse((self.root / ".failed.compile.lock").exists())

    def test_unsupported_objective_rejected_without_artifact(self):
        booster = xgb.train({"objective": "reg:absoluteerror", "nthread": 1},
                            xgb.DMatrix(self.data, label=np.arange(len(self.data))), num_boost_round=2)
        source = self.root / "unsupported.json"
        booster.save_model(source)
        target = self.root / "unsupported-artifact"
        with self.assertRaisesRegex(ValueError, "Unsupported objective"):
            compile_model(source, target, jobs=1)
        self.assertFalse(target.exists())

    def test_validation_rejects_wrong_data(self):
        with self.assertRaises(ValueError):
            validate_model(self.cases[0][1], np.zeros((3, 2)))
        report = validate_model(self.cases[0][1], self.data)
        self.assertTrue(report["passed"])
        self.assertEqual(report["checks"][1]["rows"], 100)

    def test_runtime_load_without_ml_dependencies_and_cli(self):
        artifact = str(self.cases[0][1])
        program = '''
import importlib.abc, sys
class BlockML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'xgboost', 'treelite', 'tl2cgen', 'sklearn', 'scipy'}:
            raise ImportError('Build-only dependency must not be imported: ' + fullname)
sys.meta_path.insert(0, BlockML())
from xgbfast import Predictor
with Predictor.load(sys.argv[1]) as model:
    assert isinstance(model.predict([0.0] * model.num_features), float)
print('runtime-only OK')
'''
        result = subprocess.run([sys.executable, "-c", program, artifact], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("runtime-only OK", result.stdout)
        result = subprocess.run([sys.executable, "-m", "xgbfast", "inspect", artifact], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["num_features"], 4)

    def test_direct_mode_parity_ownership_and_strict_inputs(self):
        for _, artifact, reference in self.cases:
            with Predictor.load(artifact, mode="direct") as model:
                expected = reference.inplace_predict(self.data)
                saved = model.predict(self.data)
                scalar = model.predict(self.data[0])
                self.assertIsInstance(scalar, float)
                self.assertAlmostEqual(scalar, float(expected[0]), places=6)
                missing = np.full(4, np.nan, dtype=np.float32)
                for _ in range(5):
                    np.testing.assert_allclose(model.predict(missing), reference.inplace_predict(missing.reshape(1, -1))[0], rtol=1e-5, atol=1e-6)
                    self.assertAlmostEqual(model.predict(self.data[0]), scalar, places=6)
                np.testing.assert_allclose(saved, expected, rtol=1e-5, atol=1e-6)
                out = np.empty(100, dtype=np.float32)
                self.assertIsNone(model.predict_into(self.data, out))
                np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)
                for bad in (self.data.tolist(), self.data.astype(np.float64), self.data[::2]):
                    with self.assertRaises(ValueError):
                        model.predict(bad)
                with self.assertRaises(ValueError):
                    model.predict(np.full(4, np.inf, dtype=np.float32))
                self.assertAlmostEqual(model.predict(self.data[0]), scalar, places=6)
            model.close()
            with self.assertRaises(RuntimeError):
                model.predict(self.data[0])

    def test_direct_mode_rejects_other_threads_and_reentrancy(self):
        with Predictor.load(self.cases[0][1], mode="direct") as model:
            with ThreadPoolExecutor(max_workers=1) as executor:
                for function in (lambda: model.predict(self.data[0]), model.close):
                    with self.assertRaises(RuntimeError):
                        executor.submit(function).result(timeout=5)
            original = model._native

            def reentrant(*args):
                with self.assertRaises(RuntimeError):
                    model.predict(self.data[0])
                with self.assertRaises(RuntimeError):
                    model.close()
                return original(*args)

            model._native = reentrant
            self.assertIsInstance(model.predict(self.data[0]), float)
            model._native = original
        with self.assertRaises(ValueError):
            Predictor.load(self.cases[0][1], mode="invalid")


if __name__ == "__main__":
    unittest.main()
