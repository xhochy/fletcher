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
from pandas.core.sorting import get_group_index_sorter

from ._algorithms import all_op, any_op, extract_isnull_bytemap

from typing import Union, Optional

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
    # na_value = pa.Null()

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

    @property
    def size(self):
        """
        Number of elements in this array.

        Returns
        -------
        size : int
        """
        # type: () -> int
        return len(self.data)

    @property
    def shape(self):
        # type: () -> Tuple[int]
        # This may be patched by pandas to support pseudo-2D operations.
        return (self.size,)

    @property
    def ndim(self):
        # type: () -> int
        return len(self.shape)

    def __len__(self):
        """
        Length of this array

        Returns
        -------
        length : int
        """
        # type: () -> int
        return self.shape[0]

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

    def _reduce(self, name, skipna=True, **kwargs):
        """
        Return a scalar result of performing the reduction operation.

        Parameters
        ----------
        name : str
            Name of the function, supported values are:
            { any, all, min, max, sum, mean, median, prod,
            std, var, sem, kurt, skew }.
        skipna : bool, default True
            If True, skip NaN values.
        **kwargs
            Additional keyword arguments passed to the reduction function.
            Currently, `ddof` is the only supported kwarg.

        Returns
        -------
        scalar

        Raises
        ------
        TypeError : subclass does not define reductions
        """
        if name == "any" and pa.types.is_boolean(self.dtype.arrow_dtype):
            return any_op(self.data, skipna=skipna)
        elif name == "all" and pa.types.is_boolean(self.dtype.arrow_dtype):
            return all_op(self.data, skipna=skipna)

        raise TypeError(
            "cannot perform {name} with type {dtype}".format(
                name=name, dtype=self.dtype
            )
        )

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
                if (
                    pa.types.is_integer(self.dtype.arrow_dtype)
                    or pa.types.is_floating(self.dtype.arrow_dtype)
                    or pa.types.is_boolean(self.dtype.arrow_dtype)
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
                if buf is not None:
                    size += buf.size
        return size

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

    def _indices_to_numpy_array(self, indices):
        length = len(self)
        if (isinstance(indices, slice) or np.any(indices < 0) or
                (isinstance(indices, np.ndarray) and indices.dtype.kind == 'b')):
            # pa.Array.take supports only positive indices
            indices = np.arange(length)[indices]
        else:
            indices = np.asarray(indices, dtype=np.int)
        return indices

    def take(self, indices: Union[np.array, list, slice], allow_fill=False, fill_value=True):
        # type: (Sequence[int], bool, Optional[Any]) -> ExtensionArray
        """
        Take elements from an array.

        Parameters
        ----------
        indices : sequence of integers,array of integers or slice
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
            fill_value must be given if allow_fill is true

        Returns
        -------
        FletcherArray

        Raises
        ------
        IndexError
            When the indices are out of bounds for the array.
        ValueError
            When `indices` contains negative values other than ``-1``
            and `allow_fill` is True.

        See Also
        --------
        numpy.take
        pandas.api.extensions.take
        """
        if isinstance(indices, list):
            indices = np.asarray(indices)
            if not np.any(indices < 0):
                allow_fill = False
        if isinstance(indices,slice) and allow_fill:
            Raise Exception('allow_fill cannot be used with slice')

        if not allow_fill:
            indices = self._indices_to_numpy_array(indices)
            lengths = np.fromiter(map(len, self.data.chunks), dtype=np.int64)
            cum_lengths = lengths.cumsum()

            bins = np.searchsorted(cum_lengths, indices, side='right')  #one might want to parallel this operation
            cum_lengths -= lengths
            limits_idx = np.concatenate([[0], np.bincount(bins, minlength=self.data.num_chunks).cumsum()])

            if is_increasing(bins):
                sort_idx = None
            else:
                sort_idx = get_group_index_sorter(bins, self.data.num_chunks)
                del bins
                indices = indices[sort_idx]
                sort_idx = np.argsort(sort_idx, kind="merge")  # inverse sort indices

            def take_in_one_chunk(i_chunk):
                array_idx = indices[limits_idx[i_chunk]:limits_idx[i_chunk + 1]] - cum_lengths[i_chunk]
                return self.data.chunk(i_chunk).take(pa.array(array_idx))#this is a pa.Array

            result = take_in_one_chunk(0) if self.data.num_chunks == 1 else [take_in_one_chunk(i) for i in range(self.data.num_chunks)]

            if sort_idx is None:
                return FletcherArray(pa.chunked_array(result))
            else:
                return FletcherArray(pa.concat_arrays(result).take(pa.array(sort_idx)))

        if allow_fill:
            if fill_value is None:
                raise Exception('fill_value must be given')
            error_mask = indices < -1
            if np.any(error_mask):
                raise Exception('Since allow_fill is True, there cannot be any indices < -1')
            return self._concat_same_type([self, FletcherArray([fill_value], dtype=self.data.type)]).take(indices)


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

@staticmethod
def is_increasing(array: Union[np.ndarray, list]) -> bool:
    """
    Checks if an array is sorted in increasing order
    >>> is_increasing([4, 4, 5])
    True
    >>> is_increasing([4, 3, 5])
    False
    >>> is_increasing([4, 5, 9])
    True
    >>> is_increasing(['a', 'a', 'c'])
    True
    >>> is_increasing(np.array(['2017-03-31', '2017-03-31', '2017-04-05'], dtype='datetime64[D]'))
    True
    """
    array = np.asarray(array)
    return array.size == 0 or np.all(array[1:] >= array[:-1])




