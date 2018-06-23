# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from collections import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api.types import is_array_like, is_bool_dtype, is_integer, is_integer_dtype
from pandas.core.arrays import ExtensionArray

from ._algorithms import extract_isnull_bytemap


class FletcherArrayBase(ExtensionArray):
    _can_hold_na = True

    def __init__(self, array):
        if is_array_like(array) or isinstance(array, list):
            self.data = pa.chunked_array([pa.array(array, self.dtype.arrow_dtype)])
        elif isinstance(array, self.arrow_array_type):
            self.data = pa.chunked_array([array])
        elif isinstance(array, (pa.ChunkedArray, pa.NullArray)):
            self.data = array
        else:
            raise ValueError(
                "Unsupported type passed for {}: {}".format(self.__name__, type(array))
            )

    def __array__(self):
        """
        Correctly construct numpy arrays when passed to `np.asarray()`.
        """
        return pa.column("dummy", self.data).to_pandas().values

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
        return cls(
            pa.chunked_array(
                [array for ea in to_concat for array in ea.data.iterchunks()]
            )
        )

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
        # Workaround for Arrow bug that segfaults on empty slice.
        # This is fixed in Arrow master, will be released in 0.10
        if isinstance(item, slice):
            start = item.start or 0
            stop = item.stop if item.stop is not None else len(self.data)
            stop = min(stop, len(self.data))
            # Arrow can't handle slices with steps other than 1
            # https://issues.apache.org/jira/browse/ARROW-2714
            if item.step != 1:
                return type(self)(np.asarray(self)[item])
            if stop - start == 0:
                return type(self)(pa.array([], type=self.data.type))
        elif isinstance(item, Iterable):
            if not is_array_like(item):
                item = np.array(item)
            if is_integer_dtype(item):
                return self.take(item)
            elif is_bool_dtype(item):
                indices = np.array(item)
                indices = np.argwhere(indices).flatten()
                return self.take(indices)
            else:
                raise IndexError(
                    "Only integers, slices and integer or boolean arrays are valid indices."
                )
        elif is_integer(item):
            if item < 0:
                item += len(self)
            if item >= len(self):
                return None
        value = self.data[item]
        if isinstance(value, pa.ChunkedArray):
            return type(self)(value)
        else:
            return value.as_py()

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

    def factorize(self, na_sentinel=-1):
        # type: (int) -> Tuple[ndarray, ExtensionArray]
        """Encode the extension array as an enumerated type.
        Parameters
        ----------
        na_sentinel : int, default -1
            Value to use in the `labels` array to indicate missing values.
        Returns
        -------
        labels : ndarray
            An integer NumPy array that's an indexer into the original
            ExtensionArray.
        uniques : ExtensionArray
            An ExtensionArray containing the unique values of `self`.
            .. note::
               uniques will *not* contain an entry for the NA value of
               the ExtensionArray if there are any missing values present
               in `self`.
        See Also
        --------
        pandas.factorize : Top-level factorize method that dispatches here.
        Notes
        -----
        :meth:`pandas.factorize` offers a `sort` keyword as well.
        """
        if pa.types.is_dictionary(self.data.type):
            raise NotImplementedError()
        elif self.data.num_chunks == 1:
            # Dictionaryencode and do the same as above
            encoded = self.data.chunk(0).dictionary_encode()
            indices = encoded.indices.to_pandas()
            if indices.dtype.kind == "f":
                indices[np.isnan(indices)] = na_sentinel
                indices = indices.astype(int)
            return indices, type(self)(encoded.dictionary)
        else:
            np_array = pa.column("dummy", self.data).to_pandas().values
            return pd.factorize(np_array, na_sentinel=na_sentinel)

    def astype(self, dtype, copy=True):
        """
        Cast to a NumPy array with 'dtype'.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.

        Returns
        -------
        array : ndarray
            NumPy ndarray with 'dtype' for its dtype.
        """
        if isinstance(dtype, pa.DataType):
            raise NotImplementedError("Cast propagation in astype not yet implemented")
        else:
            dtype = np.dtype(dtype)
            return np.asarray(self).astype(dtype)

    @classmethod
    def _from_sequence(cls, scalars):
        """
        Construct a new ExtensionArray from a sequence of scalars.

        Parameters
        ----------
        scalars : Sequence
            Each element will be an instance of the scalar type for this
            array, ``cls.dtype.type``.

        Returns
        -------
        ExtensionArray
        """
        return cls(pa.array(scalars))

    def take(self, indices, allow_fill=False, fill_value=None):
        # type: (Sequence[int], bool, Optional[Any]) -> ExtensionArray
        """
        Take elements from an array.

        Parameters
        ----------
        indices : sequence of integers
            Indices to be taken.
        allow_fill : bool, default False
            How to handle negative values in `indices`.
            * False: negative values in `indices` indicate positional indices
              from the right (the default). This is similar to
              :func:`numpy.take`.
            * True: negative values in `indices` indicate
              missing values. These values are set to `fill_value`. Any other
              other negative values raise a ``ValueError``.
        fill_value : any, optional
            Fill value to use for NA-indices when `allow_fill` is True.
            This may be ``None``, in which case the default NA value for
            the type, ``self.dtype.na_value``, is used.
            For many ExtensionArrays, there will be two representations of
            `fill_value`: a user-facing "boxed" scalar, and a low-level
            physical NA value. `fill_value` should be the user-facing version,
            and the implementation should handle translating that to the
            physical version for processing the take if nescessary.

        Returns
        -------
        ExtensionArray

        Raises
        ------
        IndexError
            When the indices are out of bounds for the array.
        ValueError
            When `indices` contains negative values other than ``-1``
            and `allow_fill` is True.

        Notes
        -----
        ExtensionArray.take is called by ``Series.__getitem__``, ``.loc``,
        ``iloc``, when `indices` is a sequence of values. Additionally,
        it's called by :meth:`Series.reindex`, or any other method
        that causes realignemnt, with a `fill_value`.

        See Also
        --------
        numpy.take
        pandas.api.extensions.take
        """
        from pandas.core.algorithms import take

        data = self.astype(object)
        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value
        # fill value should always be translated from the scalar
        # type for the array, to the physical storage type for
        # the data, before passing to take.
        result = take(data, indices, fill_value=fill_value, allow_fill=allow_fill)
        return self._from_sequence(result)
