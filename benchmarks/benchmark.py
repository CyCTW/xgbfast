#!/usr/bin/env python3
"""Compare Python-call latency for one XGBoost model across CPU backends."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Set before loading native runtimes; API thread counts are also set explicitly.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import tl2cgen
import treelite
import xgboost as xgb
from sklearn.datasets import make_classification
from threadpoolctl import threadpool_info, threadpool_limits
from direct import DirectPredictor, compile_predictor


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="XGBoost JSON/UBJ; otherwise train demo")
    parser.add_argument("--data", type=Path, help="2D numeric .npy input with model feature order")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--batch-sizes", type=positive_int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--iterations", type=positive_int, default=3000, help="calls per round/backend/batch")
    parser.add_argument("--rounds", type=positive_int, default=3)
    parser.add_argument("--warmup", type=positive_int, default=200)
    parser.add_argument("--pool-size", type=positive_int, default=64, help="distinct batches rotated through")
    parser.add_argument("--trees", type=positive_int, default=100)
    parser.add_argument("--depth", type=positive_int, default=4)
    parser.add_argument("--features", type=positive_int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--toolchain", default="clang" if sys.platform == "darwin" else "gcc")
    args = parser.parse_args()
    if args.model and not args.data:
        parser.error("--model requires --data so input distribution represents your model")
    if args.data and not args.model:
        parser.error("--data requires --model")
    if not args.model and args.features < 4:
        parser.error("demo requires --features >= 4")
    args.batch_sizes = list(dict.fromkeys(args.batch_sizes))
    return args


def load_or_train(args):
    if args.model:
        booster = xgb.Booster()
        booster.load_model(args.model)
        data = np.ascontiguousarray(np.load(args.data, allow_pickle=False), dtype=np.float32)
    else:
        data, labels = make_classification(
            n_samples=15000,  # Keep the trained model independent of benchmark batch sizes.
            n_features=args.features, n_informative=max(2, args.features // 2),
            n_redundant=0, random_state=args.seed,
        )
        data = np.ascontiguousarray(data, dtype=np.float32)
        demo = xgb.XGBClassifier(
            n_estimators=args.trees, max_depth=args.depth, learning_rate=0.1,
            objective="binary:logistic", tree_method="hist", device="cpu",
            n_jobs=1, random_state=args.seed,
        )
        demo.fit(data[:5000], labels[:5000])
        booster = demo.get_booster()
        data = data[5000:].copy()

    config = json.loads(booster.save_config())["learner"]
    objective = config["objective"]["name"]
    if config["gradient_booster"]["name"] != "gbtree":
        raise ValueError("This benchmark supports gbtree models only")
    if objective not in {"binary:logistic", "reg:squarederror"}:
        raise ValueError(f"Unsupported objective {objective}; use binary:logistic or reg:squarederror")
    if int(config["learner_model_param"].get("num_target", 1)) != 1:
        raise ValueError("Multi-target models are not supported")
    if data.ndim != 2 or data.shape[1] != booster.num_features():
        raise ValueError(f"Expected input shape (N, {booster.num_features()}), got {data.shape}")
    if data.shape[0] < max(args.batch_sizes):
        raise ValueError("Input needs at least as many rows as the largest batch")
    if np.isinf(data).any():
        raise ValueError("Input contains infinity; NaN missing values are supported")
    original_rounds = booster.num_boosted_rounds()
    best = booster.attr("best_iteration")
    if best is not None:
        booster = booster[:int(best) + 1]
    booster.set_param({"nthread": 1, "device": "cpu"})
    model_path = args.output / "model.json"
    booster.save_model(model_path)
    estimator = xgb.XGBClassifier(n_jobs=1) if objective == "binary:logistic" else xgb.XGBRegressor(n_jobs=1)
    estimator.load_model(model_path)
    estimator.set_params(n_jobs=1, device="cpu")
    np.save(args.output / "input.npy", data, allow_pickle=False)
    return estimator, data, {
        "objective": objective, "features": booster.num_features(),
        "trees": len(booster.get_dump()), "original_rounds": original_rounds,
        "effective_rounds": booster.num_boosted_rounds(), "best_iteration": best,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256((args.output / "input.npy").read_bytes()).hexdigest(),
        "input_shape": list(data.shape), "demo": args.model is None,
    }


def make_backends(estimator, tl_model, predictor, direct=None):
    booster = estimator.get_booster()
    iteration_range = (0, booster.num_boosted_rounds())
    if isinstance(estimator, xgb.XGBClassifier):
        def sklearn_predict(data):
            return estimator.predict_proba(data, iteration_range=iteration_range)[:, 1]
    else:
        def sklearn_predict(data):
            return estimator.predict(data, iteration_range=iteration_range)

    backends = {
        "xgb_sklearn": sklearn_predict,
        "xgb_inplace": lambda data: booster.inplace_predict(data, iteration_range=iteration_range),
        "treelite_gtil": lambda data: treelite.gtil.predict(tl_model, data, nthread=1).reshape(-1),
        "tl2cgen_numpy": lambda data: predictor.predict(tl2cgen.DMatrix(data)).reshape(-1),
        "tl2cgen_prebuilt": lambda matrix: predictor.predict(matrix).reshape(-1),
    }
    if direct is not None:
        backends["tl2cgen_direct"] = direct
    return backends


def verify(backends, data):
    # Verify every row used as the source of timed batches, plus missing values.
    missing = data[:min(32, len(data))].copy()
    missing[:, ::3] = np.nan
    missing[0, :] = np.nan
    errors = {name: 0.0 for name in backends}
    for source in (data, missing):
        for offset in range(0, len(source), 1024):
            chunk = source[offset:offset + 1024]
            reference = backends["xgb_sklearn"](chunk)
            for name, predict in backends.items():
                if name == "tl2cgen_direct":
                    predict.prepare(len(chunk))
                argument = tl2cgen.DMatrix(chunk) if name == "tl2cgen_prebuilt" else chunk
                actual = predict(argument)
                np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-6, err_msg=name)
                errors[name] = max(errors[name], float(np.max(np.abs(actual - reference))))
    return errors


def measure(predict, inputs, indices):
    samples = np.empty(len(indices), dtype=np.int64)
    clock = time.perf_counter_ns
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for position, index in enumerate(indices):
            argument = inputs[index]  # Python pool lookup is outside timed region.
            start = clock()
            predict(argument)
            samples[position] = clock() - start
    finally:
        if gc_was_enabled:
            gc.enable()
    return samples


def stats(samples, batch_size):
    us = samples / 1000.0
    return {
        "calls": len(samples), "p50_us": float(np.percentile(us, 50)),
        "p95_us": float(np.percentile(us, 95)), "p99_us": float(np.percentile(us, 99)),
        "mean_us": float(np.mean(us)), "min_us": float(np.min(us)),
        "max_us": float(np.max(us)), "mean_us_per_row": float(np.mean(us) / batch_size),
    }


def main():
    args = arguments()
    # Each run gets its own directory, so prior results/model artifacts survive.
    args.output = args.output.resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    args.output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=1):
        estimator, data, model_info = load_or_train(args)
        tl_model = treelite.frontend.load_xgboost_model(str(args.output / "model.json"))
        print("Compiling the same model with TL2cgen...", flush=True)
        start = time.perf_counter()
        lib_path = compile_predictor(tl_model, args.output / "generated", args.toolchain)
        compile_seconds = time.perf_counter() - start
        predictor = tl2cgen.Predictor(lib_path, nthread=1)
        direct = DirectPredictor(lib_path)
        backends = make_backends(estimator, tl_model, predictor, direct)
        errors = verify(backends, data)
        print(f"Prediction parity passed; max abs error = {max(errors.values()):.3g}", flush=True)

        rng = np.random.default_rng(args.seed)
        raw, summary, per_round, orders = {}, [], [], []
        for batch_size in args.batch_sizes:
            direct.prepare(batch_size)
            batches = [np.ascontiguousarray(data[rng.choice(len(data), batch_size, replace=False)])
                       for _ in range(args.pool_size)]
            matrices = [tl2cgen.DMatrix(batch) for batch in batches]
            by_backend = {name: [] for name in backends}
            for round_index in range(args.rounds):
                indices = rng.integers(0, args.pool_size, size=args.iterations)
                names = list(backends)
                rng.shuffle(names)
                orders.append({"batch_size": batch_size, "round": round_index + 1, "order": names.copy()})
                for name in names:
                    predict = backends[name]
                    inputs = matrices if name == "tl2cgen_prebuilt" else batches
                    for index in range(args.warmup):
                        predict(inputs[index % len(inputs)])
                    samples = measure(predict, inputs, indices)
                    by_backend[name].append(samples)
                    raw[f"b{batch_size}_{name}_r{round_index + 1}"] = samples
                    per_round.append({"batch_size": batch_size, "backend": name,
                                      "round": round_index + 1, **stats(samples, batch_size)})
            for name, chunks in by_backend.items():
                summary.append({"batch_size": batch_size, "backend": name,
                                **stats(np.concatenate(chunks), batch_size)})
            print(f"Finished batch_size={batch_size}", flush=True)

        compiler = subprocess.run([args.toolchain, "--version"], capture_output=True, text=True, check=True)
        metadata = {
            "utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(),
            "python": sys.version, "compiler": compiler.stdout.splitlines()[0],
            "versions": {name: importlib.metadata.version(name) for name in
                         ["numpy", "scipy", "scikit-learn", "xgboost", "treelite", "tl2cgen", "threadpoolctl"]},
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "model": model_info, "compile_seconds_excluded": compile_seconds,
            "direct": {
                "input_copy_nan_conversion_included": True,
                "python_input_checks_and_pointer_lookup_included": True,
                "output_buffer_reused": True,
                "output_ownership": "borrowed; overwritten on next call with same batch size",
                "shared_library": str(lib_path),
                "source_sha256": hashlib.sha256(Path(__file__).with_name("direct.c").read_bytes()).hexdigest(),
            },
            "max_abs_errors": errors, "threadpools": threadpool_info(),
            "timer": vars(time.get_clock_info("perf_counter")), "orders": orders,
            "summary": summary, "per_round": per_round,
        }
        direct.close()
    np.savez_compressed(args.output / "latency_ns.npz", **raw)
    (args.output / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (args.output / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print("\nLatency per Python call, microseconds (us)")
    print(f"{'batch':>5}  {'backend':<20} {'p50':>10} {'p95':>10} {'p99':>10} {'p50 speedup':>12}")
    for row in summary:
        base = next(r["p50_us"] for r in summary if r["batch_size"] == row["batch_size"] and r["backend"] == "xgb_sklearn")
        print(f"{row['batch_size']:>5}  {row['backend']:<20} {row['p50_us']:10.3f} "
              f"{row['p95_us']:10.3f} {row['p99_us']:10.3f} {base / row['p50_us']:11.2f}x")
    print(f"\nArtifacts: {args.output}")


if __name__ == "__main__":
    main()
