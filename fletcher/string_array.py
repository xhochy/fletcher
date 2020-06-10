from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa

from ._algorithms import _endswith, _startswith
from ._numba_compat import NumbaString, NumbaStringArray
from .algorithms.string import _text_cat, _text_cat_chunked, _text_cat_chunked_mixed
from .base import FletcherBaseArray, FletcherChunkedArray, FletcherContinuousArray


@pd.api.extensions.register_series_accessor("fr_text")
@pd.api.extensions.register_series_accessor("text")
class TextAccessor:
    """Accessor for pandas exposed as ``.str``."""

    def __init__(self, obj):
        if not isinstance(obj.values, FletcherBaseArray):
            raise AttributeError(
                "only Fletcher{Continuous,Chunked}Array[string] has text accessor"
            )
        self.obj = obj
        self.data = self.obj.values.data

    def cat(self, others: Optional[FletcherBaseArray]) -> pd.Series:
        """
        Concatenate strings in the Series/Index with given separator.

        If `others` is specified, this function concatenates the Series/Index
        and elements of `others` element-wise.
        If `others` is not passed, then all values in the Series/Index are
        concatenated into a single string with a given `sep`.
        """
        if not isinstance(others, pd.Series):
            raise NotImplementedError(
                "other needs to be Series of Fletcher{Chunked,Continuous}Array"
            )
        elif isinstance(others.values, FletcherChunkedArray):
            return pd.Series(
                FletcherChunkedArray(_text_cat_chunked(self.data, others.values.data))
            )
        elif not isinstance(others.values, FletcherContinuousArray):
            raise NotImplementedError("other needs to be FletcherContinuousArray")

        if isinstance(self.obj.values, FletcherChunkedArray):
            return pd.Series(
                FletcherChunkedArray(
                    _text_cat_chunked_mixed(self.data, others.values.data)
                )
            )
        else:  # FletcherContinuousArray
            return pd.Series(
                FletcherContinuousArray(_text_cat(self.data, others.values.data))
            )

    def _call_str_accessor(self, func, *args, **kwargs) -> pd.Series:
        pd_series = self.data.to_pandas()
        result = pa.array(getattr(pd_series.str, func)(*args, **kwargs).values)
        return pd.Series(type(self.obj)(result))

    def zfill(self, width: int) -> pd.Series:
        """Pad strings in the Series/Index by prepending '0' characters."""
        return self._call_str_accessor("zfill", width)

    def startswith(self, needle):
        """Check whether a row starts with a certain pattern."""
        return self._call_x_with(_startswith, needle)

    def endswith(self, needle):
        """Check whether a row ends with a certain pattern."""
        return self._call_x_with(_endswith, needle)

    def _call_x_with(self, impl, needle, na=None):
        needle = NumbaString.make(needle)  # type: ignore
        result = np.zeros(len(self.data), dtype=np.uint8)

        if isinstance(self.data, pa.ChunkedArray):
            offset = 0
            for chunk in self.data.chunks:
                str_arr = NumbaStringArray.make(chunk)  # type: ignore
                impl(str_arr, needle, 2, offset, result)
                offset += len(chunk)
        else:
            str_arr = NumbaStringArray.make(self.data)  # type: ignore
            impl(str_arr, needle, 2, 0, result)

        return pd.Series(
            type(self.obj.values)(pa.array(result.astype(bool), mask=(result == 2)))
        )
