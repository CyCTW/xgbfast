"""Optional build-time dependencies; never imported by Predictor.load()."""
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .artifact import ABI_VERSION, FORMAT_VERSION, host_target, inspect_model, sha256
from .runtime import Predictor


def _dependencies():
    try:
        import xgboost
        import treelite
        import tl2cgen
    except ImportError as error:
        raise RuntimeError('Compilation/validation requires: pip install "xgbfast[compiler]"') from error
    return xgboost, treelite, tl2cgen


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def _load_source(path, xgb):
    if path.suffix.lower() not in {".json", ".ubj"}:
        raise ValueError("Provide an XGBoost save_model() JSON or UBJ file")
    booster = xgb.Booster()
    booster.load_model(path)
    config = json.loads(booster.save_config())["learner"]
    objective = config["objective"]["name"]
    if config["gradient_booster"]["name"] != "gbtree":
        raise ValueError("Version 0.1 supports gbtree models only")
    if objective not in {"binary:logistic", "reg:squarederror"}:
        raise ValueError(f"Unsupported objective: {objective}; use binary:logistic or reg:squarederror")
    if int(config["learner_model_param"].get("num_target", 1)) != 1:
        raise ValueError("Multi-target models are not supported yet")
    if "c" in (booster.feature_types or []):
        raise ValueError("Categorical features are not supported yet; supply a numeric-feature model")
    original_rounds = booster.num_boosted_rounds()
    best = booster.attr("best_iteration")
    if best is not None:
        if not 0 <= int(best) < original_rounds:
            raise ValueError("Model contains an invalid best_iteration")
        booster = booster[:int(best) + 1]
    booster.set_param({"nthread": 1, "device": "cpu"})
    return booster, {
        "objective": objective, "num_features": booster.num_features(),
        "feature_names": booster.feature_names, "dtype": "float32",
        "output_kind": "positive_class_probability" if objective == "binary:logistic" else "regression_value",
        "input_shapes": ["(features,)", "(rows, features)"],
        "output_shapes": {"single": "float", "batch": "(rows,)"},
        "original_rounds": original_rounds, "effective_rounds": booster.num_boosted_rounds(),
        "best_iteration": None if best is None else int(best), "trees": len(booster.get_dump()),
    }


def _probes(booster):
    """Deterministic smoke probes, including split boundaries and missing values."""
    features = booster.num_features()
    rng = np.random.default_rng(42)
    random = rng.normal(size=(64, features)).astype(np.float32)
    missing = random[:16].copy()
    missing[:, ::2] = np.nan
    boundary = []
    names = booster.feature_names or [f"f{i}" for i in range(features)]
    indices = {name: index for index, name in enumerate(names)}

    def visit(node):
        if "split" not in node or len(boundary) >= 96:
            return
        feature = indices[node["split"]]
        threshold = np.float32(node["split_condition"])
        for value in (np.nextafter(threshold, np.float32(-np.inf)), threshold,
                      np.nextafter(threshold, np.float32(np.inf))):
            if np.isfinite(value):
                row = np.zeros(features, dtype=np.float32)
                row[feature] = value
                boundary.append(row)
        for child in node.get("children", []):
            visit(child)

    for tree in booster.get_dump(dump_format="json"):
        visit(json.loads(tree))
        if len(boundary) >= 96:
            break
    parts = [random, missing, np.zeros((1, features), dtype=np.float32),
             np.full((1, features), np.nan, dtype=np.float32)]
    if boundary:
        parts.append(np.asarray(boundary))
    return np.concatenate(parts)


def validate_model(artifact, data=None):
    xgb, _, _ = _dependencies()
    manifest = inspect_model(artifact)
    booster = xgb.Booster()
    booster.load_model(Path(artifact) / manifest["model"]["file"])
    booster.set_param({"nthread": 1, "device": "cpu"})
    sources = [("synthetic_smoke", _probes(booster))]
    if data is not None:
        supplied = np.load(data, allow_pickle=False) if isinstance(data, (str, Path)) else data
        supplied = np.require(supplied, dtype=np.float32, requirements=["C", "A"])
        if supplied.ndim != 2 or supplied.shape[1] != manifest["num_features"] or not len(supplied):
            raise ValueError(f"Validation data must have nonempty shape (N, {manifest['num_features']})")
        if np.isinf(supplied).any():
            raise ValueError("Validation data contains infinity")
        sources.append(("representative_data", supplied))
    results = []
    with Predictor.load(artifact) as predictor:
        for label, values in sources:
            max_error = 0.0
            for offset in range(0, len(values), 1024):
                batch = values[offset:offset + 1024]
                expected = booster.inplace_predict(batch)
                actual = predictor.predict(batch)
                if not np.isfinite(expected).all() or not np.isfinite(actual).all():
                    raise ValueError("Validation produced non-finite model outputs")
                try:
                    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
                except AssertionError as error:
                    raise ValueError(f"Prediction parity failed on {label}: {error}") from error
                max_error = max(max_error, float(np.max(np.abs(actual - expected))))
            results.append({"source": label, "rows": len(values), "max_abs_error": max_error})
    return {"passed": True, "rtol": 1e-5, "atol": 1e-6, "checks": results,
            "note": "Synthetic probes are a smoke check, not a proof for all possible inputs",
            "utc": datetime.now(timezone.utc).isoformat()}


def compile_model(model_path, output, *, validation_data=None, toolchain=None, jobs=4):
    xgb, treelite, tl2cgen = _dependencies()
    if platform.system() not in {"Darwin", "Linux"}:
        raise ValueError("Version 0.1 builds on macOS and Linux only")
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    model_path, output = Path(model_path).resolve(), Path(output).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    toolchain = toolchain or ("clang" if platform.system() == "Darwin" else "gcc")
    compiler_path = shutil.which(toolchain)
    if not compiler_path:
        raise RuntimeError(f"C compiler '{toolchain}' not found; install clang/gcc or pass --toolchain")
    # TL2cgen 1.0 constructs shell commands with this field. Restrict it to a
    # plain executable path rather than accepting shell syntax or extra flags.
    if not re.fullmatch(r"[A-Za-z0-9_./+\-]+", compiler_path):
        raise ValueError("Compiler path must not contain spaces or shell metacharacters")
    compiler_version = subprocess.run([compiler_path, "--version"], check=True, text=True,
                                      capture_output=True).stdout.splitlines()[0]
    versions = {name: importlib.metadata.version(name) for name in ["xgbfast", "xgboost", "treelite", "tl2cgen", "numpy"]}
    # Explicit libm linkage and symbol binding for standalone Linux .so loading.
    link_options = ["-lm", "-Wl,-z,defs", "-Wl,-Bsymbolic-functions"] if platform.system() == "Linux" else []
    adapter = Path(__file__).with_name("adapter.c")
    fingerprint = {"source_sha256": sha256(model_path), "adapter_sha256": sha256(adapter),
                   "builder_sha256": sha256(__file__), "versions": versions,
                   "target": host_target(), "compiler": compiler_version,
                   "jobs": jobs, "abi_version": ABI_VERSION, "link_options": link_options}
    cache_key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f".{output.name}.compile.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"Another build holds {lock}; if a previous build crashed, remove the stale lock") from error
    os.close(descriptor)
    staging = None
    try:
        if output.exists():
            old = inspect_model(output)
            if old.get("cache_key") != cache_key:
                raise FileExistsError("Output already contains a different build; choose a new output directory")
            # Verify reference-model parity on cache hits too, including new supplied data.
            validate_model(output, validation_data)
            return output
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.build-", dir=output.parent))
        booster, info = _load_source(model_path, xgb)
        booster.save_model(staging / "model.json")
        tl_model = treelite.frontend.load_xgboost_model(str(staging / "model.json"))
        generated = staging / "generated"
        tl2cgen.generate_c_code(tl_model, dirpath=generated, params={"parallel_comp": jobs})
        header = (generated / "header.h").read_text()
        if ("void predict(union Entry* data, int pred_margin, float* result);" not in header
                or "float fvalue;" not in header
                or not re.search(r"#define N_TARGET 1\s*\n", header)
                or not re.search(r"#define MAX_N_CLASS 1\s*\n", header)):
            raise ValueError("Unsupported generated C ABI; expected scalar float32 TL2cgen 1.0 output")
        shutil.copyfile(adapter, generated / "adapter.c")
        recipe_path = generated / "recipe.json"
        recipe = json.loads(recipe_path.read_text())
        recipe["sources"].append({"name": "adapter", "length": len(adapter.read_text().splitlines())})
        _write_json(recipe_path, recipe)
        library = Path(tl2cgen.create_shared(compiler_path, generated, nthread=jobs, options=link_options))
        library_name = "model" + library.suffix
        shutil.copyfile(library, staging / library_name)
        manifest = {"format_version": FORMAT_VERSION, "abi_version": ABI_VERSION,
                    **info, "target": host_target(), "cache_key": cache_key,
                    "build": fingerprint, "utc": datetime.now(timezone.utc).isoformat(),
                    "library": {"file": library_name, "sha256": sha256(staging / library_name)},
                    "model": {"file": "model.json", "sha256": sha256(staging / "model.json")}}
        _write_json(staging / "manifest.json", manifest)
        validation = validate_model(staging, validation_data)
        _write_json(staging / "validation.json", validation)
        (staging / "build.log").write_text(f"Compiler: {compiler_version}\nCommand backend: TL2cgen create_shared\nBuild and validation passed.\n")
        # Publish only a complete, validated artifact. Never replace another build.
        if output.exists():
            raise FileExistsError(f"Output appeared during build: {output}")
        staging.rename(output)
        staging = None
        return output
    except Exception as error:
        if staging is not None:
            logs = [f"{type(error).__name__}: {error}"]
            for path in staging.rglob("*.log"):
                logs.append(path.read_text(errors="replace")[-100000:])
            (output.parent / f"{output.name}.build-error.log").write_text("\n".join(logs))
        raise
    finally:
        if staging is not None:
            shutil.rmtree(staging)  # Only our uncommitted temporary build directory.
        lock.unlink()
