# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from ._algorithms import extract_isnull_bytemap
from pandas.core.arrays import ExtensionArray

import pyarrow as pa


class FletcherArrayBase(ExtensionArray):
    _can_hold_na = True

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
            return type(self)(value)
        else:
            return value

    def isna(self):
        # type: () -> np.ndarray
        """
        Boolean NumPy array indicating if each value is missing.

        This should return a 1-D array the same length as 'self'.
        """
        return extract_isnull_bytemap(self.data)

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
        return type(self)(self.data)

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
