from typing import Any, Optional

import numpy as np
import pandas as pd

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

    def zfill(self, width: int) -> pd.Series:
        """Pad strings in the Series/Index by prepending '0' characters."""
        # TODO: This will extend all strings to be at least width wide but we need to take uncode into account where the length could be smaller due to multibyte characters
        # This will require a StringBuilder class or a run where we pre-compute the size of the final array
        raise NotImplementedError("zfill")

    def startswith(self, needle, na=None):
        """Check whether a row starts with a certain pattern."""
        return self._call_x_with(_startswith, needle, na)

    def endswith(self, needle, na=None):
        """Check whether a row ends with a certain pattern."""
        return self._call_x_with(_endswith, needle, na)

    def _call_x_with(self, impl, needle, na=None):
        needle = NumbaString.make(needle)  # type: ignore

        if isinstance(na, bool):
            result = np.zeros(len(self.data), dtype=np.bool)
            na_arg: Any = np.bool_(na)

        else:
            result = np.zeros(len(self.data), dtype=np.uint8)
            na_arg = 2

        offset = 0
        for chunk in self.data.chunks:
            str_arr = NumbaStringArray.make(chunk)  # type: ignore
            impl(str_arr, needle, na_arg, offset, result)
            offset += len(chunk)

        result = pd.Series(result, index=self.obj.index, name=self.obj.name)
        return (
            result if isinstance(na, bool) else result.map({0: False, 1: True, 2: na})
        )
