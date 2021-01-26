import math
import types
from typing import Optional, Union

import numba
import numba.experimental
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from pandas.core.strings import StringMethods

from fletcher._algorithms import _extract_isnull_bitmap
from fletcher.algorithms.bool import all_true_like
from fletcher.algorithms.string import (
    _endswith,
    _slice_handle_chunk,
    _startswith,
    _text_cat,
    _text_cat_chunked,
    _text_cat_chunked_mixed,
    _text_contains_case_sensitive,
    _text_count_case_sensitive,
    _text_replace_case_sensitive,
    _text_strip,
)
from fletcher.base import (
    FletcherBaseArray,
    FletcherChunkedArray,
    FletcherContinuousArray,
)

# Do we support calling the `.str.` accessor on fletcher arrays?
SUPPORTS_STR_ON_FLETCHER = any(
    "StringArrayMethods" in str(x) for x in FletcherBaseArray.mro()
)


def buffers_as_arrays(sa):
    buffers = sa.buffers()
    return (
        _extract_isnull_bitmap(sa, 0, len(sa)),
        np.asarray(buffers[1]).view(np.uint32),
        np.asarray(buffers[2]).view(np.uint8),
    )


@numba.experimental.jitclass(
    [
        ("missing", numba.uint8[:]),
        ("offsets", numba.uint32[:]),
        ("data", numba.optional(numba.uint8[:])),
        ("offset", numba.int64),
    ]
)
class NumbaStringArray:
    """Wrapper around arrow's StringArray for use in numba functions.

    Usage::

        NumbaStringArray.make(array)
    """

    def __init__(self, missing, offsets, data, offset):
        self.missing = missing
        self.offsets = offsets
        self.data = data
        self.offset = offset

    @property
    def byte_size(self):
        # TODO: offset?
        return self.data.shape[0]

    @property
    def size(self):
        return len(self.offsets) - 1 - self.offset

    def isnull(self, str_idx):
        str_idx += self.offset
        byte_idx = str_idx // 8
        bit_mask = 1 << (str_idx % 8)
        return (self.missing[byte_idx] & bit_mask) == 0

    def byte_length(self, str_idx):
        str_idx += self.offset
        return self.offsets[str_idx + 1] - self.offsets[str_idx]

    def get_byte(self, str_idx, byte_idx):
        str_idx += self.offset
        full_idx = self.offsets[str_idx] + byte_idx
        return self.data[full_idx]

    def length(self, str_idx):
        result = 0
        byte_length = self.byte_length(str_idx)
        current = 0

        while current < byte_length:
            _, inc = self.get(str_idx, current)
            current += inc
            result += 1

        return result

    # TODO: implement this
    def get(self, str_idx, byte_idx):
        b = self.get_byte(str_idx, byte_idx)
        if b > 127:
            raise ValueError()

        return b, 1

    def decode(self, str_idx):
        byte_length = self.byte_length(str_idx)
        buffer = np.zeros(byte_length, np.int32)

        i = 0
        j = 0
        while i < byte_length:
            code, inc = self.get(str_idx, i)
            buffer[j] = code

            i += inc
            j += 1

        return buffer[:j]


def _make(cls, sa):
    if not isinstance(sa, pa.StringArray):
        sa = pa.array(sa, pa.string())

    return cls(*buffers_as_arrays(sa), offset=sa.offset)


# @classmethod does not seem to be supported
NumbaStringArray.make = types.MethodType(_make, NumbaStringArray)  # type: ignore


@numba.experimental.jitclass(
    [("start", numba.uint32), ("end", numba.uint32), ("data", numba.uint8[:])]
)
class NumbaString:
    def __init__(self, data, start=0, end=None):
        if end is None:
            end = data.shape[0]

        self.data = data
        self.start = start
        self.end = end

    @property
    def length(self):
        return self.end - self.start

    def get_byte(self, i):
        return self.data[self.start + i]


def _make_string(cls, obj):
    if isinstance(obj, str):
        data = obj.encode("utf8")
        data = np.asarray(memoryview(data))

        return cls(data, 0, len(data))

    raise TypeError()


NumbaString.make = types.MethodType(_make_string, NumbaString)  # type: ignore


@numba.experimental.jitclass(
    [
        ("missing", numba.uint8[:]),
        ("offsets", numba.uint32[:]),
        ("data", numba.optional(numba.uint8[:])),
        ("string_position", numba.uint32),
        ("byte_position", numba.uint32),
        ("string_capacity", numba.uint32),
        ("byte_capacity", numba.uint32),
    ]
)
class NumbaStringArrayBuilder:
    def __init__(self, string_capacity, byte_capacity):
        self.missing = np.ones(_missing_capactiy(string_capacity), np.uint8)
        self.offsets = np.zeros(string_capacity + 1, np.uint32)
        self.data = np.zeros(byte_capacity, np.uint8)
        self.string_position = 0
        self.byte_position = 0

        self.string_capacity = string_capacity
        self.byte_capacity = byte_capacity

    def increase_string_capacity(self, string_capacity):
        assert string_capacity > self.string_capacity

        missing = np.zeros(_missing_capactiy(string_capacity), np.uint8)
        missing[: _missing_capactiy(self.string_capacity)] = self.missing
        self.missing = missing

        offsets = np.zeros(string_capacity + 1, np.uint32)
        offsets[: self.string_capacity + 1] = self.offsets
        self.offsets = offsets

        self.string_capacity = string_capacity

    def increase_byte_capacity(self, byte_capacity):
        assert byte_capacity > self.byte_capacity

        data = np.zeros(byte_capacity, np.uint8)
        data[: self.byte_capacity] = self.data
        self.data = data

        self.byte_capacity = byte_capacity

    def put_byte(self, b):
        if self.byte_position >= self.byte_capacity:
            self.increase_byte_capacity(int(math.ceil(1.2 * self.byte_capacity)))

        self.data[self.byte_position] = b
        self.byte_position += 1

    def finish_string(self):
        if self.string_position >= self.string_capacity:
            self.increase_string_capacity(int(math.ceil(1.2 * self.string_capacity)))

        self.offsets[self.string_position + 1] = self.byte_position

        byte_idx = self.string_position // 8
        self.missing[byte_idx] |= 1 << (self.string_position % 8)

        self.string_position += 1

    def finish_null(self):
        if self.string_position >= self.string_capacity:
            self.increase_string_capacity(int(math.ceil(1.2 * self.string_capacity)))

        self.offsets[self.string_position + 1] = self.byte_position

        byte_idx = self.string_position // 8
        self.missing[byte_idx] &= ~(1 << (self.string_position % 8))

        self.string_position += 1

    def finish(self):
        self.missing = self.missing[: _missing_capactiy(self.string_position)]
        self.offsets = self.offsets[: self.string_position + 1]
        self.data = self.data[: self.byte_position]


@numba.jit
def _missing_capactiy(capacity):
    return int(math.ceil(capacity / 8))


class TextAccessorBase:
    """Base class for ``.fr_str`` and ``.fr_strx`` accessors."""

    def __init__(self, obj):
        self.obj = obj
        self.data = self.obj.values.data

    def _series_like(self, array: Union[pa.Array, pa.ChunkedArray]) -> pd.Series:
        """Return an Arrow result as a series with the same base classes as the input."""
        return pd.Series(
            type(self.obj.values)(array),
            dtype=type(self.obj.dtype)(array.type),
            index=self.obj.index,
        )

    def _call_str_accessor(self, func, *args, **kwargs) -> pd.Series:
        """Call the str accessor function with transforming the Arrow series to pandas series
        and back."""
        pd_series = self.data.to_pandas()
        pd_series.index = self.obj.index
        pd_result = getattr(pd_series.str, func)(*args, **kwargs)
        if isinstance(pd_result, pd.DataFrame):
            for c in pd_result.columns:
                pd_result[c] = type(self.obj.values)(
                    pd_result[c].replace({np.nan: None}).values
                )
            return pd_result
        elif isinstance(pd_result, pd.Series):
            array = pa.array(pd_result.values, from_pandas=True)
            return self._series_like(array)
        else:
            raise AttributeError(f"{func} returned unexpected type {type(pd_result)}")

    def _wrap_str_accessor(self, func):
        """Return a str accessor function that includes the transformation from Arrow series
        to pandas series and back."""

        def _wrapped_str_accessor(*args, **kwargs) -> pd.Series:
            return self._call_str_accessor(func, *args, **kwargs)

        return _wrapped_str_accessor

    @staticmethod
    def _validate_str_accessor(func):
        """Raise an exception if the given function name is not a valid function of StringMethods."""
        if not (
            hasattr(pd.core.strings.StringMethods, func)
            and callable(getattr(pd.core.strings.StringMethods, func))
        ):
            raise AttributeError(
                f"{func} not available in pd.core.strings.StringMethods nor in fletcher.string_array.TextAccessor"
            )


@pd.api.extensions.register_series_accessor("fr_str")
class TextAccessorExt(TextAccessorBase):
    """Accessor for pandas exposed as ``.fr_str``."""

    def __init__(self, obj):
        """Accessor for pandas exposed as ``.fr_str``.
        fletcher functionality will be used if available otherwise str functions are invoked."""
        if not isinstance(obj.values, FletcherBaseArray):
            # call StringMethods to validate the input obj
            StringMethods(obj)
        super().__init__(obj)

    def __getattr__(self, name):
        TextAccessorBase._validate_str_accessor(name)
        if isinstance(self.obj.values, FletcherBaseArray):
            if hasattr(TextAccessor, name) and callable(getattr(TextAccessor, name)):
                return getattr(TextAccessor(self.obj), name)
            return self._wrap_str_accessor(name)
        return getattr(self.obj.str, name)


@pd.api.extensions.register_series_accessor("fr_strx")
class TextAccessor(TextAccessorBase):
    """Accessor for pandas exposed as ``.fr_strx``."""

    def __init__(self, obj):
        if not isinstance(obj.values, FletcherBaseArray):
            raise AttributeError(
                "only Fletcher{Continuous,Chunked}Array[string] has text accessor"
            )
        super().__init__(obj)

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

    def contains(self, pat: str, case: bool = True, regex: bool = True) -> pd.Series:
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
                contains_exact = getattr(
                    pc, "match_substring", _text_contains_case_sensitive
                )
                # Can just check for a match on the byte-sequence
                return self._series_like(contains_exact(self.data, pat))
            else:
                # Check if pat is all-ascii, then use lookup-table for lowercasing
                # else: use libutf8proc
                pass
        return self._call_str_accessor("contains", pat=pat, case=case, regex=regex)

    def count(self, pat: str, regex: bool = True) -> pd.Series:
        if not regex:
            return self._series_like(_text_count_case_sensitive(self.data, pat))
        return self._call_str_accessor("count", pat=pat)

    def replace(
        self, pat: str, repl: str, n: int = -1, case: bool = True, regex: bool = True
    ):
        """
        Replace occurrences of pattern/regex in the Series/Index with some other string.
        Equivalent to str.replace() or re.sub().

        Return а string Series where in each row the occurrences of the given
        pattern or regex ``pat`` are replaced by ``repl``.

        This implementation differs to the one in ``pandas``:
         * We always return a missing for missing data.
         * You cannot pass flags for the regular expression module.

        Parameters
        ----------
        pat : str
            Character sequence or regular expression.
        repl : str
            Replacement string.
        n : int
            Number of replacements to make from start.
        case : bool, default True
            If True, case sensitive.
        regex : bool, default True
            If True, assumes the pat is a regular expression.
            If False, treats the pat as a literal string.

        Returns
        -------
        Series of string values.
        """
        if n == 0:
            return self._series_like(self.data)
        if not regex:
            if case:
                return self._series_like(
                    _text_replace_case_sensitive(self.data, pat, repl, n)
                )
        return self._call_str_accessor(
            "replace", pat=pat, repl=repl, n=n, case=case, regex=regex
        )

    def strip(self, to_strip=None):
        """Strip whitespaces from both ends of strings."""
        # see for unicode spaces: https://en.wikibooks.org/wiki/Unicode/Character_reference/2000-2FFF
        # for whatever reason 0x200B (zero width space) is not considered a space by pandas.split()
        if to_strip is None:
            to_strip = (
                " \t\r\n\x85\x1f\x1e\x1d\x1c\x0c\x0b\xa0"
                "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2000\u2009\u200A\u2028\u2029\u202F\u205F"
            )
        return self._series_like(_text_strip(self.data, to_strip))

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

    def slice(self, start=0, end=None, step=1):
        """Extract every `step` character from strings from `start` to `end`."""
        return self._series_like(_slice_handle_chunk(self.data, start, end, step))


if SUPPORTS_STR_ON_FLETCHER:
    TextAccessor.isalnum = lambda self: self.obj.str.isalnum()  # type: ignore
    TextAccessor.isalpha = lambda self: self.obj.str.isalpha()  # type: ignore
    TextAccessor.isdigit = lambda self: self.obj.str.isdigit()  # type: ignore
    TextAccessor.isspace = lambda self: self.obj.str.isspace()  # type: ignore
    TextAccessor.islower = lambda self: self.obj.str.islower()  # type: ignore
    TextAccessor.isupper = lambda self: self.obj.str.isupper()  # type: ignore
    TextAccessor.istitle = lambda self: self.obj.str.istitle()  # type: ignore
    TextAccessor.isnumeric = lambda self: self.obj.str.isnumeric()  # type: ignore
    TextAccessor.isdecimal = lambda self: self.obj.str.isdecimal()  # type: ignore
elif hasattr(pc, "utf8_is_alnum"):
    TextAccessor.isalnum = lambda self: self._series_like(  # type: ignore
        pc.utf8_is_alnum(self.data)
    )
    TextAccessor.isalpha = lambda self: self._series_like(  # type: ignore
        pc.utf8_is_alpha(self.data)
    )
    TextAccessor.isdigit = lambda self: self._series_like(  # type: ignore
        pc.utf8_is_digit(self.data)
    )
    TextAccessor.islower = lambda self: self._series_like(  # type: ignore
        pc.utf8_is_lower(self.data)
    )
    TextAccessor.isupper = lambda self: self._series_like(  # type: ignore
        pc.utf8_is_upper(self.data)
    )
    TextAccessor.istitle = lambda self: self._series_like(  # type: ignore
        pc.utf8_is_title(self.data)
    )
    TextAccessor.isnumeric = lambda self: self._series_like(  # type: ignore
        pc.utf8_is_numeric(self.data)
    )
    TextAccessor.isdecimal = lambda self: self._series_like(  # type: ignore
        pc.utf8_is_decimal(self.data)
    )
    # This one was added later
    if hasattr(pc, "utf8_is_space"):
        TextAccessor.isspace = lambda self: self._series_like(  # type: ignore
            pc.utf8_is_space(self.data)
        )
