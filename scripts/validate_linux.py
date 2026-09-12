"""Build, check and benchmark on a NATIVE Linux host; no emulation required."""
import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/linux"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--cpu", type=int, help="Optional allowed Linux CPU id for benchmark affinity")
    parser.add_argument("--max-direct-ratio", type=float, help="Optional direct/legacy p50 acceptance bound")
    parser.add_argument("--iterations", type=int, default=10000)
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Run this script on native Linux; macOS results do not validate Linux latency")
    if bool(args.model) != bool(args.data):
        parser.error("Provide both --model and --data, or neither for a demo")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    repo = Path(__file__).resolve().parents[1]
    # Resolve caller-relative paths before subprocess cwd changes.
    args.output = args.output.resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    args.output.mkdir(parents=True)
    if args.model:
        args.model, args.data = args.model.resolve(), args.data.resolve()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
                    str(repo / "tests"), "-v"], check=True)
    if args.model is None:
        import numpy as np
        import xgboost as xgb
        from sklearn.datasets import make_classification
        data, labels = make_classification(n_samples=15000, n_features=32,
                                           n_informative=16, n_redundant=0, random_state=42)
        data = np.ascontiguousarray(data, dtype=np.float32)
        model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                  objective="binary:logistic", tree_method="hist", device="cpu",
                                  n_jobs=1, random_state=42)
        model.fit(data[:5000], labels[:5000])
        args.model, args.data = args.output / "source.json", args.output / "input.npy"
        model.save_model(args.model)
        np.save(args.data, data[5000:])
    from xgbfast import compile_model, inspect_model
    artifact = compile_model(args.model, args.output / "model", validation_data=args.data)
    manifest = inspect_model(artifact)
    dependencies = subprocess.run(["ldd", str(artifact / manifest["library"]["file"])],
                                  capture_output=True, text=True, check=True).stdout
    (args.output / "ldd.txt").write_text(dependencies)
    if "not found" in dependencies or any(name in dependencies for name in
                                          ("libxgboost", "libtreelite", "libtl2cgen", "libgomp", "libomp")):
        raise RuntimeError(f"Unexpected dependency in generated model library:\n{dependencies}")
    # A fresh process proves runtime loading does not import model-building libraries.
    program = '''
import importlib.abc, sys
class NoML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'xgboost', 'treelite', 'tl2cgen', 'sklearn', 'scipy'}:
            raise ImportError(fullname)
sys.meta_path.insert(0, NoML())
import numpy as np
from xgbfast import Predictor
for mode in ('direct', 'native'):
    with Predictor.load(sys.argv[1], mode=mode) as model:
        assert isinstance(model.predict(np.zeros(model.num_features, dtype=np.float32)), float)
print('Linux direct/native runtime-only checks passed')
'''
    subprocess.run([sys.executable, "-c", program, str(artifact)], check=True)
    command = [sys.executable, str(repo / "benchmarks/compare.py"),
               str(artifact), "--data", str(args.data), "--output", str(args.output / "benchmark"),
               "--iterations", str(args.iterations)]
    if args.cpu is not None:
        command += ["--cpu", str(args.cpu)]
    if args.max_direct_ratio is not None:
        command += ["--max-direct-ratio", str(args.max_direct_ratio)]
    subprocess.run(command, check=True)
    (args.output / "validation-status.json").write_text(json.dumps({
        "passed": True, "host": platform.platform(), "machine": platform.machine(),
        "artifact": str(artifact), "benchmark_command": command}, indent=2))
    print(f"Linux validation completed: {args.output}")


if __name__ == "__main__":
    main()
