FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work
COPY pyproject.toml setup.py README.md requirements-bench.txt ./
COPY src ./src
COPY tests ./tests
COPY scripts ./scripts
COPY benchmarks ./benchmarks
RUN python scripts/install_bench.py && python -m pip install .
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ENTRYPOINT ["python", "scripts/validate_linux.py"]
CMD ["--output", "/results"]
