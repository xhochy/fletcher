from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa

from ._algorithms import _endswith, _startswith, all_true_like
from ._numba_compat import NumbaString, NumbaStringArray
from .algorithms.string import (
    _text_cat,
    _text_cat_chunked,
    _text_cat_chunked_mixed,
    _text_contains_case_sensitive,
)
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
        return self._series_like(
            pa.array(getattr(pd_series.str, func)(*args, **kwargs).values)
        )

    def _series_like(self, array):
        """Return an Arrow result as a series with the same base classes as the input."""
        return pd.Series(
            type(self.obj.values)(array),
            dtype=type(self.obj.dtype)(array.type),
            index=self.obj.index,
        )

    def contains(self, pat, case=True, regex=True):
        """
        Test if pattern or regex is contained within a string of a Series or Index.

        Return boolean Series or Index based on whether a given pattern or regex is
        contained within a string of a Series or Index.

        This implementation differs to the one in ``pandas``:
         * We always return a missing for missing data.
         * You cannot pass flags for the regular expression module.

        Parameters
        ----------
        pat : str
            Character sequence or regular expression.
        case : bool, default True
            If True, case sensitive.
        regex : bool, default True
            If True, assumes the pat is a regular expression.

            If False, treats the pat as a literal string.

        Returns
        -------
        Series or Index of boolean values
            A Series or Index of boolean values indicating whether the
            given pattern is contained within the string of each element
            of the Series or Index.
        """
        if not regex:
            if len(pat) == 0:
                # For an empty pattern return all-True array
                return self._series_like(all_true_like(self.data))

            if case:
                # Can just check for a match on the byte-sequence
                return self._series_like(_text_contains_case_sensitive(self.data, pat))
            else:
                # Check if pat is all-ascii, then use lookup-table for lowercasing
                # else: use libutf8proc
                pass
        return self._call_str_accessor("contains", pat=pat, case=case, regex=regex)

    def zfill(self, width: int) -> pd.Series:
        """Pad strings in the Series/Index by prepending '0' characters."""
        return self._call_str_accessor("zfill", width)

    def startswith(self, pat):
        """Check whether a row starts with a certain pattern."""
        return self._call_x_with(_startswith, pat)

    def endswith(self, pat):
        """Check whether a row ends with a certain pattern."""
        return self._call_x_with(_endswith, pat)

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
