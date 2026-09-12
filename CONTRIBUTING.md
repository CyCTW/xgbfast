# Development

Run all commands from this standalone repository's root. Follow README.md for
Linux compiler and Python environment setup, then:

```bash
python scripts/install_bench.py
python -m pip install -e .
python -m unittest discover -s tests -v
```

Reinstall after changing C++ headers or native.cpp to rebuild the extension.
Tests compile small classification and regression models and a standalone C++
consumer. They require a compiler and the benchmark dependencies.

Build a distributable wheel:

```bash
python -m pip wheel --no-deps . --wheel-dir dist
```

Validate on the target architecture with `python scripts/validate_linux.py`.
Do not compare absolute microsecond timings across unrelated hosts or runs.
Keep generated models, virtual environments, and raw measurements out of Git.

Source layout:

- `src/xgbfast`: compiler, artifact validation, Python APIs, C adapter, CPython extension.
- `src/xgbfast/include`: public C ABI and C++20 wrapper.
- `examples`: Python and C++ consumers.
- `tests`: prediction parity, input contracts, resource lifecycle, C++ integration.
- `benchmarks`: self-contained TL2cgen baselines and xgbfast comparison.
- `scripts`: dependency installation and Linux validation.

The generated model ABI is versioned independently of the Python package.
Preserve compatibility or explicitly increment the ABI and update the loaders.
