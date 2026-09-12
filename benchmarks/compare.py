"""Re-run all backends on one model, including the old direct prototype.

Standalone benchmark; install requirements-bench.txt first.
All inputs start at prepared NumPy float32 features; feature engineering excluded.
"""
import argparse
import csv
import importlib.metadata
import json
import platform
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import benchmark as previous  # Sets thread limits before native library use.
import direct as legacy
import numpy as np
import tl2cgen
import treelite
import xgboost as xgb
from threadpoolctl import threadpool_info, threadpool_limits
from xgbfast import Predictor, inspect_model
from xgbfast.artifact import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--iterations", type=previous.positive_int, default=10000)
    parser.add_argument("--rounds", type=previous.positive_int, default=3)
    parser.add_argument("--warmup", type=previous.positive_int, default=300)
    parser.add_argument("--cpu", type=int, help="Linux CPU to pin this process to")
    parser.add_argument("--max-direct-ratio", type=float,
                        help="Optional acceptance limit: direct-mode p50 / legacy-direct p50")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = parser.parse_args()
    if args.cpu is not None:
        if not hasattr(os, "sched_setaffinity"):
            parser.error("--cpu requires Linux sched_setaffinity")
        os.sched_setaffinity(0, {args.cpu})
    if args.max_direct_ratio is not None and args.max_direct_ratio <= 0:
        parser.error("--max-direct-ratio must be positive")
    args.output = args.output.resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    args.output.mkdir(parents=True)
    manifest = inspect_model(args.artifact)
    data = np.ascontiguousarray(np.load(args.data, allow_pickle=False), dtype=np.float32)
    if data.ndim != 2 or len(data) < 64 or data.shape[1] != manifest["num_features"]:
        raise ValueError("Need at least 64 data rows matching the model's features")
    model_path = args.artifact / manifest["model"]["file"]
    wrapper_type = xgb.XGBClassifier if manifest["objective"] == "binary:logistic" else xgb.XGBRegressor
    estimator = wrapper_type(n_jobs=1)
    estimator.load_model(model_path)
    estimator.set_params(n_jobs=1, device="cpu")
    tree = treelite.frontend.load_xgboost_model(str(model_path))
    with (threadpool_limits(limits=1), Predictor.load(args.artifact) as model,
          Predictor.load(args.artifact, mode="direct") as fast,
          Predictor.load(args.artifact, mode="native") as native):
        legacy_lib = legacy.compile_predictor(tree, args.output / "legacy-generated",
                                              "clang" if sys.platform == "darwin" else "gcc")
        direct = legacy.DirectPredictor(legacy_lib)
        predictor = tl2cgen.Predictor(args.artifact / manifest["library"]["file"], nthread=1)
        backends = previous.make_backends(estimator, tree, predictor, direct)
        previous.verify(backends, data)
        direct.prepare(1)
        backends["legacy_direct"] = backends.pop("tl2cgen_direct")
        backends["xgbfast_predict"] = model.predict
        backends["xgbfast_direct"] = fast.predict
        backends["xgbfast_native"] = native.predict
        out = np.empty(1, dtype=np.float32)

        def into(features):
            model.predict_into(features, out)
            return out

        backends["xgbfast_predict_into"] = into
        rng = np.random.default_rng(42)
        selected = rng.choice(len(data), 64, replace=False)
        batches = [data[index:index + 1].copy() for index in selected]
        vectors = [batch[0] for batch in batches]  # 1D views created outside timing.
        matrices = [tl2cgen.DMatrix(batch) for batch in batches]
        inputs = {name: vectors if name.startswith("xgbfast") else
                  matrices if name == "tl2cgen_prebuilt" else batches for name in backends}
        max_errors = {}
        for name, function in backends.items():
            error = 0.0
            for index in range(len(batches)):
                expected = backends["xgb_sklearn"](batches[index])
                actual = np.asarray(function(inputs[name][index])).reshape(-1)
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
                error = max(error, float(np.max(np.abs(actual - expected))))
            max_errors[name] = error
        raw, per_round, orders = {}, [], []
        accumulated = {name: [] for name in backends}
        for round_id in range(args.rounds):
            indices = rng.integers(0, len(batches), size=args.iterations)
            names = list(backends)
            rng.shuffle(names)
            orders.append(names.copy())
            for name in names:
                for index in range(args.warmup):
                    backends[name](inputs[name][index % len(batches)])
                samples = previous.measure(backends[name], inputs[name], indices)
                accumulated[name].append(samples)
                raw[f"{name}_r{round_id + 1}"] = samples
                per_round.append({"backend": name, "round": round_id + 1, **previous.stats(samples, 1)})
            print(f"Round {round_id + 1}/{args.rounds} done", flush=True)
        summary = [{"backend": name, **previous.stats(np.concatenate(samples), 1)}
                   for name, samples in accumulated.items()]
        report = {"model_sha256": manifest["model"]["sha256"], "input_sha256": sha256(args.data),
                  "utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
                  "python": sys.version, "artifact": str(args.artifact.resolve()),
                  "iterations": args.iterations, "rounds": args.rounds, "warmup": args.warmup,
                  "input_rows_rotated": 64, "threadpools": threadpool_info(),
                  "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                  "cpu_info": Path("/proc/cpuinfo").read_text() if sys.platform == "linux" else None,
                  "versions": {name: importlib.metadata.version(name) for name in
                               ["xgbfast", "numpy", "xgboost", "scikit-learn", "treelite", "tl2cgen"]},
                  "notes": ["Warm CPU single-row Python call; optional CPU pinning; no clock control",
                            "Every backend consumes the same rows in the same order within each round",
                            "xgbfast_predict: 1D NumPy to owned Python float, all API checks included",
                            "xgbfast_direct: thread-confined, strict float32, owned Python float, same C core",
                            "xgbfast_native: CPython extension + shared C++ wrapper; GIL held, float32 buffer to owned float",
                            "xgbfast_predict_into: 1D NumPy to caller-owned float32 output; checks included",
                            "legacy_direct: 2D NumPy to borrowed internal array",
                            "Inputs start after feature engineering; C feature copy/NaN handling included"],
                  "orders": orders, "max_abs_errors": max_errors, "per_round": per_round, "summary": summary}
        by_name = {row["backend"]: row for row in summary}
        ratio = by_name["xgbfast_direct"]["p50_us"] / by_name["legacy_direct"]["p50_us"]
        report["direct_vs_legacy_p50_ratio"] = ratio
        report["acceptance_limit"] = args.max_direct_ratio
        report["acceptance_passed"] = None if args.max_direct_ratio is None else ratio <= args.max_direct_ratio
        direct.close()
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output / "latency_ns.npz", **raw)
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = ["| Backend | p50 µs | p95 µs | p99 µs |", "|---|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['backend']} | {row['p50_us']:.3f} | {row['p95_us']:.3f} | {row['p99_us']:.3f} |")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nArtifacts: {args.output}")
    print(f"Direct-mode / legacy-direct p50 ratio: {ratio:.3f}")
    if report["acceptance_passed"] is False:
        raise SystemExit(f"Direct-mode ratio {ratio:.3f} exceeded {args.max_direct_ratio}; raw results retained")


if __name__ == "__main__":
    main()
