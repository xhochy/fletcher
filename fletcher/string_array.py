# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import datetime
import numpy as np
import pandas as pd
import pyarrow as pa
import six
from pandas.api.types import is_array_like
from pandas.core.dtypes.dtypes import ExtensionDtype

from ._algorithms import _endswith, _startswith
from ._numba_compat import NumbaString, NumbaStringArray
from .base import FletcherArrayBase


_python_type_map = {pa.date64().id: datetime.date, pa.string().id: six.text_type}

_string_type_map = {"date64[ms]": pa.date64(), "string": pa.string()}


class FletcherDtype(ExtensionDtype):

    def __init__(self, arrow_dtype):
        self.arrow_dtype = arrow_dtype

    def __str__(self):
        return str(self.arrow_dtype)

    def __repr__(self):
        return 'FletcherDType({})'.format(str(self))

    def __eq__(self, other):
        """Check whether 'other' is equal to self.
        By default, 'other' is considered equal if
        * it's a string matching 'self.name'.
        * it's an instance of this type.
        Parameters
        ----------
        other : Any
        Returns
        -------
        bool
        """
        if isinstance(other, six.string_types):
            return other == self.name
        elif isinstance(other, type(self)):
            return self.arrow_dtype == other.arrow_dtype
        else:
            return False

    @property
    def type(self):
        # type: () -> type
        """The scalar type for the array, e.g. ``int``
        It's expected ``ExtensionArray[item]`` returns an instance
        of ``ExtensionDtype.type`` for scalar ``item``.
        """
        return _python_type_map[self.arrow_dtype.id]

    @property
    def kind(self):
        # type () -> str
        """A character code (one of 'biufcmMOSUV'), default 'O'
        This should match the NumPy dtype used when the array is
        converted to an ndarray, which is probably 'O' for object if
        the extension type cannot be represented as a built-in NumPy
        type.
        See Also
        --------
        numpy.dtype.kind
        """
        return self.arrow_dtype.to_pandas_dtype().char

    @property
    def name(self):
        # type: () -> str
        """A string identifying the data type.
        Will be used for display in, e.g. ``Series.dtype``
        """
        return str(self.arrow_dtype)

    @classmethod
    def construct_from_string(cls, string):
        """Attempt to construct this type from a string.
        Parameters
        ----------
        string : str
        Returns
        -------
        self : instance of 'cls'
        Raises
        ------
        TypeError
            If a class cannot be constructed from this 'string'.
        Examples
        --------
        If the extension dtype can be constructed without any arguments,
        the following may be an adequate implementation.
        >>> @classmethod
        ... def construct_from_string(cls, string)
        ...     if string == cls.name:
        ...         return cls()
        ...     else:
        ...         raise TypeError("Cannot construct a '{}' from "
        ...                         "'{}'".format(cls, string))
        """
        if string in _string_type_map:
            return cls(_string_type_map[string])
        else:
            raise TypeError("Cannot construct a '{}' from " "'{}'".format(cls, string))


class StringArray(FletcherArrayBase):
    dtype = FletcherDtype(pa.string())

    def __init__(self, array):
        if is_array_like(array) or isinstance(array, list):
            self.data = pa.chunked_array([pa.array(array, pa.string())])
        elif isinstance(array, pa.StringArray):
            self.data = pa.chunked_array([array])
        elif isinstance(array, (pa.ChunkedArray, pa.NullArray)):
            self.data = array
        else:
            raise ValueError(
                "Unsupported type passed for StringArray: {}".format(type(array))
            )


class Date64Array(FletcherArrayBase):
    dtype = FletcherDtype(pa.date64())

    def __init__(self, array):
        if is_array_like(array) or isinstance(array, list):
            self.data = pa.chunked_array([pa.array(array, pa.date64())])
        elif isinstance(array, pa.Date64Array):
            self.data = pa.chunked_array([array])
        elif isinstance(array, pa.Date32Array):
            self.data = pa.chunked_array([array.cast(pa.date64())])
        elif isinstance(array, (pa.ChunkedArray, pa.NullArray)):
            self.data = array
        else:
            raise ValueError(
                "Unsupported type passed for Date64Array: {}".format(type(array))
            )


@pd.api.extensions.register_series_accessor("text")
class TextAccessor:

    def __init__(self, obj):
        if not isinstance(obj.values, StringArray):
            raise AttributeError("only StringArray has text accessor")
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
        return (
            result if isinstance(na, bool) else result.map({0: False, 1: True, 2: na})
        )
