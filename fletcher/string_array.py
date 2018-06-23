# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import pyarrow as pa
import six
from pandas.core.dtypes.dtypes import ExtensionDtype

from ._algorithms import _endswith, _startswith
from ._numba_compat import NumbaString, NumbaStringArray
from .base import FletcherArrayBase


class StringDtype(ExtensionDtype):
    name = "string"
    type = six.text_type
    kind = "O"
    arrow_dtype = pa.string()

    @classmethod
    def construct_from_string(cls, string):
        if string == "string":
            return cls()
        else:
            raise TypeError("Cannot construct a '{}' from " "'{}'".format(cls, string))


class StringArray(FletcherArrayBase):
    dtype = StringDtype()

    arrow_array_type = pa.StringArray


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
