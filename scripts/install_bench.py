"""Install pinned benchmark dependencies, including the ARM64 source-build fix."""
import os
import platform
import subprocess
import sys
from pathlib import Path


def main():
    env = os.environ.copy()
    if platform.system() == "Linux" and platform.machine() in {"aarch64", "arm64"}:
        # TL2cgen 1.0.0 fetches Treelite 4.1.2, whose postprocessor header
        # uses std::int32_t without including <cstdint>; GCC 13 rejects it.
        env["CXXFLAGS"] = (env.get("CXXFLAGS", "") + " -include cstdint").strip()
    subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                    str(Path(__file__).resolve().parents[1] / "requirements-bench.txt")],
                   env=env, check=True)


if __name__ == "__main__":
    main()
