"""
EdgeVecDB - A lightweight vector search library for edge devices
"""

__version__ = "0.1.0"

try:
    from .edgevecdb_core import *
except ImportError as e:
    raise ImportError(
        f"Failed to import EdgeVecDB C++ extension: {e}. "
        "Make sure the package was installed correctly and all dependencies are available."
    ) from e

# Import Python utilities if available
try:
    from . import kp
except ImportError:
    pass

__all__ = [
    # Core classes and functions from C++ module
    "FlatIndex",
    "MetricType", 
    "Device",
    "L2Renorm",
    # Version info
    "__version__",
]