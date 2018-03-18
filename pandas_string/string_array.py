from pandas.core.arrays import ExtensionArray
from pandas.core.dtypes.dtypes import ExtensionDtype

import numpy as np
import pandas as pd
import pyarrow as pa

from ._numba_compat import NumbaStringArray, NumbaString
from ._algorithms import _isnull, _startswith, _endswith


class StringDtypeType(object):
    """
    The type of StringDtype, this metaclass determines subclass ability
    """
    pass


class StringDtype(ExtensionDtype):
    name = 'string'
    type = StringDtypeType
    kind = 'O'

    @classmethod
    def construct_from_string(cls, string):
        if string == "string":
            return cls()
        else:
            raise TypeError("Cannot construct a '{}' from "
                            "'{}'".format(cls, string))


class StringArray(ExtensionArray):
    dtype = StringDtype()
    _can_hold_na = True

    def __init__(self, array):
        if isinstance(array, list):
            self.data = pa.chunked_array([pa.array(array, pa.string())])
        elif isinstance(array, pa.StringArray):
            self.data = pa.chunked_array([array])
        elif isinstance(array, pa.ChunkedArray):
            self.data = array
        else:
            raise ValueError("Unsupported type passed for StringArray: {}".format(type(array)))

    def __len__(self):
        """
        Length of this array

        Returns
        -------
        length : int
        """
        # type: () -> int
        return len(self.data)

    @classmethod
    def _concat_same_type(cls, to_concat):
        # type: (Sequence[ExtensionArray]) -> ExtensionArray
        """
        Concatenate multiple array

        Parameters
        ----------
        to_concat : sequence of this type

        Returns
        -------
        ExtensionArray
        """
        return cls(pa.chunked_array([array for ea in to_concat for array in ea.data.iterchunks()]))

    def __getitem__(self, item):
        # type (Any) -> Any
        """Select a subset of self.
        Parameters
        ----------
        item : int, slice, or ndarray
            * int: The position in 'self' to get.
            * slice: A slice object, where 'start', 'stop', and 'step' are
              integers or None
            * ndarray: A 1-d boolean NumPy ndarray the same length as 'self'
        Returns
        -------
        item : scalar or ExtensionArray
        Notes
        -----
        For scalar ``item``, return a scalar value suitable for the array's
        type. This should be an instance of ``self.dtype.type``.
        For slice ``key``, return an instance of ``ExtensionArray``, even
        if the slice is length 0 or 1.
        For a boolean mask, return an instance of ``ExtensionArray``, filtered
        to the values where ``item`` is True.
        """
        value = self.data[item]
        if isinstance(value, pa.ChunkedArray):
            return StringArray(value)
        else:
            return value

    def isna(self):
        # type: () -> np.ndarray
        """
        Boolean NumPy array indicating if each value is missing.

        This should return a 1-D array the same length as 'self'.
        """
        # TODO: We should be able to return the valid bitmap as bytemap here
        result = np.zeros(len(self.data), dtype=bool)

        offset = 0
        for chunk in self.data.chunks:
            _isnull(NumbaStringArray.make(chunk), offset, result)
            offset += len(chunk)

        return result

    def copy(self, deep=False):
        # type: (bool) -> ExtensionArray
        """
        Return a copy of the array.

        Parameters
        ----------
        deep : bool, default False
            Also copy the underlying data backing this array.

        Returns
        -------
        ExtensionArray
        """
        if deep:
            raise NotImplementedError("Deep copy is not supported")
        return StringArray(self.data)

    @property
    def nbytes(self):
        # type: () -> int
        """
        The number of bytes needed to store this object in memory.
        """
        size = 0
        for chunk in self.data.chunks:
            for buf in chunk.buffers():
                size += buf.size
        return size

    @property
    def size(self):
        """
        return the number of elements in the underlying data
        """
        return len(self.data)

    @property
    def base(self):
        """
        the base object of the underlying data
        """
        return self.data


@pd.api.extensions.register_series_accessor("text")
class TextAccessor:
    def __init__(self, obj):
        if not isinstance(obj.values, StringArray):
            raise AttributeError('only StringArray has text accessor')
        self.obj = obj
        self.data = self.obj.values.data

    def startswith(self, needle, na=None):
        return self._call_x_with(_startswith, needle, na)

    def endswith(self, needle, na=None):
        return self._call_x_with(_endswith, needle, na)

    def _call_x_with(self, impl, needle, na=None):
        needle = NumbaString.make(needle)

        if isinstance(na, bool):
            result = np.zeros(len(self.data), dtype=np.bool)
            na_arg = np.bool_(na)

        else:
            result = np.zeros(len(self.data), dtype=np.uint8)
            na_arg = 2

        offset = 0
        for chunk in self.data.chunks:
            impl(NumbaStringArray.make(chunk), needle, na_arg, offset, result)
            offset += len(chunk)

        result = pd.Series(result, index=self.obj.index, name=self.obj.name)
        return result if isinstance(na, bool) else result.map({0: False, 1: True, 2: na})
