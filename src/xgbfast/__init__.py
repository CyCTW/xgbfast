"""The runtime imports only NumPy and the Python standard library."""
from .artifact import CompatibilityError, inspect_model
from .runtime import Predictor

__version__ = "0.3.0"
__all__ = ["Predictor", "compile_model", "validate_model", "inspect_model", "CompatibilityError", "get_include"]


def get_include():
    """Installed C/C++ header directory; no ML compiler imports."""
    from pathlib import Path
    return str(Path(__file__).with_name("include"))


def compile_model(model_path, output, *, validation_data=None, toolchain=None, jobs=4):
    from .compiler import compile_model as compile_impl
    return compile_impl(model_path, output, validation_data=validation_data,
                        toolchain=toolchain, jobs=jobs)


def validate_model(artifact, data=None):
    from .compiler import validate_model as validate_impl
    return validate_impl(artifact, data)
