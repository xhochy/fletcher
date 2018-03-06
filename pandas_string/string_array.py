from pandas.core.arrays import ExtensionArray
from pandas.core.dtypes.dtypes import ExtensionDtype

import numpy as np
import pandas as pd
import pyarrow as pa


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
        if isinstance(array, pa.StringArray):
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
        for i in range(len(self.data)):
            if self.data[i].as_py() is None:
                result[i] = True
        return result
