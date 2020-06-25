import pkg_resources

from ._compat import HAS_DASK
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

if HAS_DASK:
    import fletcher._dask_compat  # noqa: F401

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
