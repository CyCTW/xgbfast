# Standalone validation

2026-09-12 UTC, Ubuntu 24.04 ARM64 VM, CPython 3.12, GCC 13.3, glibc 2.39.

The standalone source tree was built and installed with `python -m pip install .`.
Then `python scripts/validate_linux.py --output results/standalone --cpu 0`
ran without any input model or data from the original parent project.

- All 16 tests passed, including a separately compiled C++ consumer.
- The validator generated its own fixed-seed model and 10,000 representative rows.
- Model compilation and XGBoost prediction parity passed.
- The generated library dependency check passed.
- Fresh-process direct and native runtime checks passed with ML imports blocked.
- All three benchmark rounds completed (10,000 calls per backend per round).

| Python backend | p50 µs | p95 µs | p99 µs |
|---|---:|---:|---:|
| sklearn | 46.125 | 48.458 | 52.875 |
| TL2cgen, prebuilt DMatrix | 5.375 | 5.708 | 5.958 |
| original ctypes direct | 1.416 | 1.500 | 1.541 |
| xgbfast direct | 1.708 | 1.791 | 1.834 |
| xgbfast native | 0.375 | 0.417 | 0.417 |

Warm single-row timing with CPU affinity; this is a VM observation, not a
production guarantee. Raw samples and model artifacts are local ignored outputs.
Run the command above to reproduce a fresh report. Docker and remote GitHub CI
were prepared but not executed in this session.
