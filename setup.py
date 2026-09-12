import sys
from setuptools import Extension, setup

setup(ext_modules=[Extension("xgbfast._native", ["src/xgbfast/native.cpp"],
                            language="c++", extra_compile_args=["-std=c++20", "-O3"],
                            libraries=["dl"] if sys.platform == "linux" else [])])
