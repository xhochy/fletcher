# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import datetime
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import six
from pandas.api.types import (
    is_array_like,
    is_bool_dtype,
    is_int64_dtype,
    is_integer,
    is_integer_dtype,
)
from pandas.core.arrays import ExtensionArray
from pandas.core.dtypes.dtypes import ExtensionDtype

from ._algorithms import extract_isnull_bytemap

_python_type_map = {
    pa.null().id: six.text_type,
    pa.bool_().id: bool,
    pa.int8().id: int,
    pa.uint8().id: int,
    pa.int16().id: int,
    pa.uint16().id: int,
    pa.int32().id: int,
    pa.uint32().id: int,
    pa.int64().id: int,
    pa.uint64().id: int,
    pa.float16().id: float,
    pa.float32().id: float,
    pa.float64().id: float,
    pa.date32().id: datetime.date,
    pa.date64().id: datetime.date,
    pa.timestamp("ms").id: datetime.datetime,
    pa.binary().id: six.binary_type,
    pa.string().id: six.text_type,
    # Use any list type here, only LIST is important
    pa.list_(pa.string()).id: list,
}

_string_type_map = {"date64[ms]": pa.date64(), "string": pa.string()}


class FletcherDtype(ExtensionDtype):
    def __init__(self, arrow_dtype):
        self.arrow_dtype = arrow_dtype

    def __hash__(self):
        return hash(self.arrow_dtype)

    def __str__(self):
        return "fletcher[{}]".format(self.arrow_dtype)

    def __repr__(self):
        return "FletcherDType({})".format(str(self.arrow_dtype))

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
        if pa.types.is_date(self.arrow_dtype):
            return "O"
        else:
            return np.dtype(self.arrow_dtype.to_pandas_dtype()).kind

    @property
    def name(self):
        # type: () -> str
        """A string identifying the data type.
        Will be used for display in, e.g. ``Series.dtype``
        """
        return str(self)

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
        # Remove fletcher specific naming from the arrow type string.
        if string.startswith("fletcher["):
            string = string[9:-1]

        if string == "list<item: string>":
            return cls(pa.list_(pa.string()))

        try:
            type_for_alias = pa.type_for_alias(string)
        except (ValueError, KeyError):
            # pandas API expects a TypeError
            raise TypeError(string)

        return cls(type_for_alias)

    @classmethod
    def construct_array_type(cls, *args):
        """
        Return the array type associated with this dtype

        Returns
        -------
        type
        """
        if len(args) > 0:
            raise NotImplementedError("construct_array_type does not support arguments")
        return FletcherArray


class FletcherArray(ExtensionArray):
    _can_hold_na = True

    def __init__(self, array, dtype=None, copy=None):
        # Copy is not used at the moment. It's only affect will be when we
        # allow array to be a FletcherArray
        if is_array_like(array) or isinstance(array, list):
            self.data = pa.chunked_array([pa.array(array, type=dtype)])
        elif isinstance(array, pa.Array):
            # TODO: Assert dtype
            self.data = pa.chunked_array([array])
        elif isinstance(array, pa.ChunkedArray):
            # TODO: Assert dtype
            self.data = array
        else:
            raise ValueError(
                "Unsupported type passed for {}: {}".format(
                    self.__class__.__name__, type(array)
                )
            )
        self._dtype = FletcherDtype(self.data.type)
        self.offsets = self._calculate_chunk_offsets()

    @property
    def dtype(self):
        # type: () -> ExtensionDtype
        return self._dtype

    def __array__(self, copy=None):
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

    def _calculate_chunk_offsets(self):
        """
        Returns an array holding the indices pointing to the first element of each chunk
        """
        offset = 0
        offsets = []
        for chunk in self.data.iterchunks():
            offsets.append(offset)
            offset += len(chunk)
        return np.array(offsets)

    def _get_chunk_indexer(self, array):
        """
        Returns an array with the chunk number for each index
        """
        if self.data.num_chunks == 1:
            return np.broadcast_to(0, len(array))
        return np.digitize(array, self.offsets[1:])

    def __setitem__(self, key, value):
        # type: (Union[int, np.ndarray], Any) -> None
        """Set one or more values inplace.

        Parameters
        ----------
        key : int, ndarray, or slice
            When called from, e.g. ``Series.__setitem__``, ``key`` will be
            one of

            * scalar int
            * ndarray of integers.
            * boolean ndarray
            * slice object

        value : ExtensionDtype.type, Sequence[ExtensionDtype.type], or object
            value or values to be set of ``key``.

        Returns
        -------
        None
        """
        # Convert all possible input key types to an array of integers
        if is_bool_dtype(key):
            key = np.argwhere(key).flatten()
        elif isinstance(key, slice):
            key = np.array(range(len(self))[key])
        elif is_integer(key):
            key = np.array([key])
        else:
            key = np.asanyarray(key)

        if pd.api.types.is_scalar(value):
            value = np.broadcast_to(value, len(key))
        else:
            value = np.asarray(value)

        if len(key) != len(value):
            raise ValueError("Length mismatch between index and value.")

        affected_chunks_index = self._get_chunk_indexer(key)
        affected_chunks_unique = np.unique(affected_chunks_index)

        all_chunks = list(self.data.iterchunks())

        for ix, offset in zip(
            affected_chunks_unique, self.offsets[affected_chunks_unique]
        ):
            chunk = all_chunks[ix]

            # Translate the array-wide indices to indices of the chunk
            key_chunk_indices = np.argwhere(affected_chunks_index == ix).flatten()
            array_chunk_indices = key[key_chunk_indices] - offset

            if pa.types.is_date64(self.dtype.arrow_dtype):
                # ARROW-2741: pa.array from np.datetime[D] and type=pa.date64 produces invalid results
                arr = np.array(chunk.to_pylist())
                arr[array_chunk_indices] = np.array(value)[key_chunk_indices]
                pa_arr = pa.array(arr, self.dtype.arrow_dtype)
            else:
                arr = chunk.to_pandas()
                # In the case where we zero-copy Arrow to Pandas conversion, the
                # the resulting arrays are read-only.
                if not arr.flags.writeable:
                    arr = arr.copy()
                arr[array_chunk_indices] = value[key_chunk_indices]

                mask = None
                # ARROW-2806: Inconsistent handling of np.nan requires adding a mask
                if pa.types.is_integer(self.dtype.arrow_dtype) or pa.types.is_floating(
                    self.dtype.arrow_dtype
                ):
                    nan_values = pd.isna(value[key_chunk_indices])
                    if any(nan_values):
                        nan_index = key_chunk_indices & nan_values
                        mask = np.ones_like(arr, dtype=bool)
                        mask[nan_index] = False
                pa_arr = pa.array(arr, self.dtype.arrow_dtype, mask=mask)
            all_chunks[ix] = pa_arr

        self.data = pa.chunked_array(all_chunks)

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
            step = item.step if item.step is not None else 1
            # Arrow can't handle slices with steps other than 1
            # https://issues.apache.org/jira/browse/ARROW-2714
            if step != 1:
                arr = np.asarray(self)[item]
                # ARROW-2806: Inconsistent handling of np.nan requires adding a mask
                if pa.types.is_integer(self.dtype.arrow_dtype) or pa.types.is_floating(
                    self.dtype.arrow_dtype
                ):
                    mask = pd.isna(arr)
                else:
                    mask = None
                return type(self)(pa.array(arr, type=self.dtype.arrow_dtype, mask=mask))
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
        # type: (int) -> Tuple[np.ndarray, ExtensionArray]
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
            if not is_int64_dtype(indices):
                indices = indices.astype(np.int64)
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
        if self.dtype == dtype:
            return self

        if isinstance(dtype, FletcherDtype):
            dtype = dtype.arrow_dtype.to_pandas_dtype()
            arrow_type = dtype.arrow_dtype
        elif isinstance(dtype, pa.DataType):
            dtype = dtype.to_pandas_dtype()
            arrow_type = dtype
        else:
            dtype = np.dtype(dtype)
            arrow_type = None
        # NumPy's conversion of list->unicode is differently from Python's
        # default. We want to have the default Python output, so force it here.
        if pa.types.is_list(self.dtype.arrow_dtype) and dtype.kind == "U":
            return np.vectorize(six.text_type)(np.asarray(self))
        if arrow_type is not None:
            return FletcherArray(np.asarray(self).astype(dtype), dtype=arrow_type)
        else:
            return np.asarray(self).astype(dtype)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=None):
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
        if isinstance(scalars, FletcherArray):
            return scalars
        if dtype and isinstance(dtype, FletcherDtype):
            dtype = dtype.arrow_dtype
        return cls(pa.array(scalars, type=dtype, from_pandas=True))

    def fillna(self, value=None, method=None, limit=None):
        """ Fill NA/NaN values using the specified method.
        Parameters
        ----------
        value : scalar, array-like
            If a scalar value is passed it is used to fill all missing values.
            Alternatively, an array-like 'value' can be given. It's expected
            that the array-like have the same length as 'self'.
        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
            Method to use for filling holes in reindexed Series
            pad / ffill: propagate last valid observation forward to next valid
            backfill / bfill: use NEXT valid observation to fill gap
        limit : int, default None
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill. In other words, if there is
            a gap with more than this number of consecutive NaNs, it will only
            be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled.
        Returns
        -------
        filled : ExtensionArray with NA/NaN filled
        """
        from pandas.api.types import is_array_like
        from pandas.util._validators import validate_fillna_kwargs
        from pandas.core.missing import pad_1d, backfill_1d

        value, method = validate_fillna_kwargs(value, method)

        mask = self.isna()

        if is_array_like(value):
            if len(value) != len(self):
                raise ValueError(
                    "Length of 'value' does not match. Got ({}) "
                    " expected {}".format(len(value), len(self))
                )
            value = value[mask]

        if mask.any():
            if method is not None:
                func = pad_1d if method == "pad" else backfill_1d
                new_values = func(self.astype(object), limit=limit, mask=mask)
                new_values = self._from_sequence(new_values, self._dtype.arrow_dtype)
            else:
                # fill with value
                new_values = self.copy()
                new_values[mask] = value
        else:
            new_values = self.copy()
        return new_values

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
        return self._from_sequence(result, dtype=self.data.type)


def pandas_from_arrow(arrow_object):
    """
    Converts Arrow object instance to their Pandas equivalent by using Fletcher.

    The conversion rules are:
      * {RecordBatch, Table} -> DataFrame
      * {Array, ChunkedArray, Column} -> Series
    """
    if isinstance(arrow_object, pa.RecordBatch):
        data = OrderedDict()
        for ix, arr in enumerate(arrow_object):
            col_name = arrow_object.schema.names[ix]
            data[col_name] = FletcherArray(arr)
        return pd.DataFrame(data)
    elif isinstance(arrow_object, pa.Table):
        data = OrderedDict()
        for col in arrow_object.itercolumns():
            data[col.name] = FletcherArray(col.data)
        return pd.DataFrame(data)
    elif isinstance(arrow_object, (pa.ChunkedArray, pa.Array)):
        return pd.Series(FletcherArray(arrow_object))
    elif isinstance(arrow_object, pa.Column):
        return pd.Series(FletcherArray(arrow_object.data), name=arrow_object.name)
    else:
        raise NotImplementedError(
            "Objects of type {} are not supported".format(type(arrow_object))
        )
