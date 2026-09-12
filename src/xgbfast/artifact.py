"""Artifact integrity and conservative host compatibility checks."""
import hashlib
import json
import platform
import sys
from pathlib import Path

ABI_VERSION = 1
FORMAT_VERSION = 1


class CompatibilityError(ValueError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def host_target():
    return {"system": platform.system(), "machine": platform.machine(),
            "byteorder": sys.byteorder,
            "macos": platform.mac_ver()[0], "libc": list(platform.libc_ver()),
            "cpu_flags": "compiler default; no -march=native"}


def _version(value):
    return tuple(int(part) for part in value.split(".") if part.isdigit())


def inspect_model(artifact, *, check_host=True):
    root = Path(artifact).resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["format_version"] != FORMAT_VERSION or manifest["abi_version"] != ABI_VERSION:
            raise CompatibilityError("Unsupported artifact format or C ABI version; recompile the model")
        if manifest["num_features"] < 1 or manifest["dtype"] != "float32":
            raise CompatibilityError("Invalid input schema")
        target = manifest["target"]
        if check_host:
            host = host_target()
            for field in ("system", "machine", "byteorder"):
                if target[field] != host[field]:
                    raise CompatibilityError(f"Artifact {field}={target[field]} differs from host {host[field]}; compile on this platform")
            if host["system"] == "Darwin" and _version(host["macos"]) < _version(target["macos"]):
                raise CompatibilityError("Artifact was built on a newer macOS; recompile on this host")
            if host["system"] == "Linux" and target["libc"][0]:
                if target["libc"][0] != host["libc"][0] or _version(host["libc"][1]) < _version(target["libc"][1]):
                    raise CompatibilityError("Artifact libc is incompatible; recompile on this host")
        for name in ("library", "model"):
            item = manifest[name]
            path = (root / item["file"]).resolve()
            if path.parent != root or sha256(path) != item["sha256"]:
                raise CompatibilityError(f"Artifact {name} path or checksum is invalid")
        return manifest
    except (KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        raise CompatibilityError(f"Cannot read valid xgbfast artifact at {root}: {error}") from error
