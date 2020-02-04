import pkg_resources

from .base import (
    FletcherBaseArray,
    FletcherBaseDtype,
    FletcherChunkedArray,
    FletcherChunkedDtype,
    FletcherContinuousArray,
    FletcherContinuousDtype,
    pandas_from_arrow,
)
from .io import read_parquet
from .string_array import TextAccessor

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:  # pragma: no cover
    __version__ = "unknown"

__all__ = [
    "FletcherBaseArray",
    "FletcherBaseDtype",
    "FletcherChunkedArray",
    "FletcherChunkedDtype",
    "FletcherContinuousArray",
    "FletcherContinuousDtype",
    "TextAccessor",
    "pandas_from_arrow",
    "read_parquet",
    "__version__",
]
