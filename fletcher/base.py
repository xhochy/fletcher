import datetime
import operator
from collections import OrderedDict
from collections.abc import Iterable
from copy import copy as copycopy
from distutils.version import LooseVersion
from functools import partialmethod
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api.types import (
    is_array_like,
    is_bool_dtype,
    is_int64_dtype,
    is_integer,
    is_integer_dtype,
)
from pandas.core.arrays import ExtensionArray
from pandas.core.dtypes.dtypes import ExtensionDtype, register_extension_dtype
from pandas.util._decorators import doc

from fletcher._algorithms import (
    extract_isnull_bytemap,
    kurt_op,
    max_op,
    median_op,
    min_op,
    np_ufunc_op,
    prod_op,
    skew_op,
    std_op,
    sum_op,
    take_on_pyarrow_list,
    var_op,
)
from fletcher.algorithms.bool import all_op, all_true, any_op, or_na, or_vectorised
from fletcher.algorithms.utils.chunking import _calculate_chunk_offsets
from fletcher.string_mixin import StringSupportingExtensionArray

PANDAS_GE_0_26_0 = LooseVersion(pd.__version__) >= "0.26.0"
if PANDAS_GE_0_26_0:
    from pandas.core.indexers import check_array_indexer

ARROW_GE_0_18_0 = LooseVersion(pa.__version__) >= "0.18.0"

_python_type_map = {
    pa.null().id: str,
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
    pa.binary().id: bytes,
    pa.string().id: str,
    # Use any list type here, only LIST is important
    pa.list_(pa.string()).id: list,
    # Use any large list type here, only LIST is important
    pa.large_list(pa.string()).id: list,
    # Use any dictionary type here, only dict is important
    pa.dictionary(pa.int32(), pa.int32()).id: dict,
    pa.duration("ns").id: datetime.timedelta,
}

_string_type_map = {"date64[ms]": pa.date64(), "string": pa.string()}

_examples = {
    pa.null(): pa.array([None, None], type=pa.null()),
    pa.bool_(): pa.array([None, True], type=pa.bool_()),
    pa.int8(): pa.array([None, -1], type=pa.int8()),
    pa.uint8(): pa.array([None, 1], type=pa.uint8()),
    pa.int16(): pa.array([None, -1], type=pa.int16()),
    pa.uint16(): pa.array([None, 1], type=pa.uint16()),
    pa.int32(): pa.array([None, -1], type=pa.int32()),
    pa.uint32(): pa.array([None, 1], type=pa.uint32()),
    pa.int64(): pa.array([None, -1], type=pa.int64()),
    pa.uint64(): pa.array([None, 1], type=pa.uint64()),
    pa.float16(): pa.array([None, np.float16(-0.1)], type=pa.float16()),
    pa.float32(): pa.array([None, -0.1], type=pa.float32()),
    pa.float64(): pa.array([None, -0.1], type=pa.float64()),
    pa.date32(): pa.array([None, datetime.date(2010, 9, 8)], type=pa.date32()),
    pa.date64(): pa.array([None, datetime.date(2010, 9, 8)], type=pa.date64()),
    pa.timestamp("s"): pa.array(
        [None, datetime.datetime(2013, 12, 11, 10, 9, 8)], type=pa.timestamp("s")
    ),
    pa.timestamp("ms"): pa.array(
        [None, datetime.datetime(2013, 12, 11, 10, 9, 8, 1000)], type=pa.timestamp("ms")
    ),
    pa.timestamp("us"): pa.array(
        [None, datetime.datetime(2013, 12, 11, 10, 9, 8, 7)], type=pa.timestamp("us")
    ),
    pa.timestamp("ns"): pa.array(
        [None, datetime.datetime(2013, 12, 11, 10, 9, 8, 7)], type=pa.timestamp("ns")
    ),
    pa.binary(): pa.array([None, b"122"], type=pa.binary()),
    pa.string(): pa.array([None, "🤔"], type=pa.string()),
    pa.duration("s"): pa.array(
        [None, datetime.timedelta(seconds=9)], type=pa.duration("s")
    ),
    pa.duration("ms"): pa.array(
        [None, datetime.timedelta(milliseconds=8)], type=pa.duration("ms")
    ),
    pa.duration("us"): pa.array(
        [None, datetime.timedelta(microseconds=7)], type=pa.duration("us")
    ),
    pa.duration("ns"): pa.array(
        [None, datetime.timedelta(microseconds=7)], type=pa.duration("ns")
    ),
}


def _get_example(arrow_dtype: pa.DataType) -> pa.Array:
    if isinstance(arrow_dtype, pa.ListType):
        return pa.array(
            [None, _get_example(arrow_dtype.value_type).to_pylist()], type=arrow_dtype
        )
    return _examples[arrow_dtype]


def _is_numeric(arrow_dtype: pa.DataType) -> bool:
    return (
        pa.types.is_integer(arrow_dtype)
        or pa.types.is_floating(arrow_dtype)
        or pa.types.is_decimal(arrow_dtype)
    )


class FletcherBaseDtype(ExtensionDtype):
    """Dtype base for a pandas ExtensionArray backed by an Apache Arrow structure."""

    na_value = pd.NA

    def __init__(self, arrow_dtype: pa.DataType):
        self.arrow_dtype = arrow_dtype

    def __hash__(self) -> int:
        """Hash the Dtype."""
        return hash(self.arrow_dtype)

    def __eq__(self, other) -> bool:
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
        if isinstance(other, str):
            return other == self.name
        elif isinstance(other, type(self)):
            return self.arrow_dtype == other.arrow_dtype
        else:
            return False

    @property
    def type(self):
        """Return the scalar type for the array, e.g. ``int``.

        It's expected ``ExtensionArray[item]`` returns an instance
        of ``ExtensionDtype.type`` for scalar ``item``.
        """
        return _python_type_map[self.arrow_dtype.id]

    @property
    def kind(self) -> str:
        """Return a character code (one of 'biufcmMOSUV'), default 'O'.

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
        elif self._is_list:
            return "O"
        elif pa.types.is_string(self.arrow_dtype):
            return "U"
        else:
            return np.dtype(self.arrow_dtype.to_pandas_dtype()).kind

    @property
    def name(self) -> str:
        """Return a string identifying the data type.

        Will be used for display in, e.g. ``Series.dtype``
        """
        return str(self)

    @property
    def _is_boolean(self):
        return pa.types.is_boolean(self.arrow_dtype)

    @property
    def _is_numeric(self):
        return _is_numeric(self.arrow_dtype)

    @property
    def _is_list(self):
        return pa.types.is_list(self.arrow_dtype) or pa.types.is_large_list(
            self.arrow_dtype
        )

    def __from_arrow__(self, data):
        """Construct a FletcherArray from an arrow array."""
        return self.construct_array_type()(data)

    def example(self):
        """Get a simple array with example content."""
        return self.construct_array_type()(_get_example(self.arrow_dtype))


@register_extension_dtype
class FletcherContinuousDtype(FletcherBaseDtype):
    """Dtype for a pandas ExtensionArray backed by Apache Arrow's pyarrow.Array."""

    def __str__(self) -> str:
        """Convert to string."""
        return f"fletcher_continuous[{self.arrow_dtype}]"

    def __repr__(self) -> str:
        """Return the textual representation of this object."""
        return "FletcherContinuousDtype({})".format(str(self.arrow_dtype))

    @classmethod
    def construct_from_string(cls, string: str):
        """Attempt to construct this type from a string.

        Parameters
        ----------
        string

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
        if not isinstance(string, str):
            raise TypeError(
                "'construct_from_string' expects a string, got <class 'int'>"
            )

        # Remove fletcher specific naming from the arrow type string.
        if string.startswith("fletcher_continuous["):
            string = string[len("fletcher_continuous[") : -1]
        else:
            raise TypeError(
                f"Cannot construct a 'FletcherContinuousDtype' from '{string}'"
            )

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
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        if len(args) > 0:
            raise NotImplementedError("construct_array_type does not support arguments")
        return FletcherContinuousArray


@register_extension_dtype
class FletcherChunkedDtype(FletcherBaseDtype):
    """Dtype for a pandas ExtensionArray backed by Apache Arrow's pyarrow.ChunkedArray."""

    def __str__(self) -> str:
        """Convert to string."""
        return f"fletcher_chunked[{self.arrow_dtype}]"

    def __repr__(self) -> str:
        """Return the textual representation of this object."""
        return "FletcherChunkedDtype({})".format(str(self.arrow_dtype))

    @classmethod
    def construct_from_string(cls, string: str) -> "FletcherChunkedDtype":
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
        if not isinstance(string, str):
            raise TypeError(
                "'construct_from_string' expects a string, got <class 'int'>"
            )

        # Remove fletcher specific naming from the arrow type string.
        if string.startswith("fletcher_chunked["):
            string = string[len("fletcher_chunked[") : -1]
        else:
            raise TypeError(
                f"Cannot construct a 'FletcherChunkedDtype' from '{string}'"
            )

        if string == "list<item: string>":
            return cls(pa.list_(pa.string()))

        try:
            type_for_alias = pa.type_for_alias(string)
        except (ValueError, KeyError):
            # pandas API expects a TypeError
            raise TypeError(string)

        return cls(type_for_alias)

    @classmethod
    def construct_array_type(cls, *args) -> "Type[FletcherChunkedArray]":
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        if len(args) > 0:
            raise NotImplementedError("construct_array_type does not support arguments")
        return FletcherChunkedArray


class FletcherBaseArray(StringSupportingExtensionArray):
    """Pandas ExtensionArray implementation base backed by an Apache Arrow structure."""

    _can_hold_na = True

    @property
    def dtype(self) -> ExtensionDtype:
        """Return the ExtensionDtype of this array."""
        return self._dtype

    def __array__(self, *args, **kwargs) -> np.ndarray:
        """Correctly construct numpy arrays when passed to `np.asarray()`."""
        return self.data.__array__(*args, **kwargs)

    def __arrow_array__(self, type=None):
        """Convert myself to a pyarrow Array or ChunkedArray."""
        return self.data

    @property
    def size(self) -> int:
        """
        Return the number of elements in this array.

        Returns
        -------
        size : int
        """
        return len(self.data)

    @property
    def shape(self) -> Tuple[int]:
        """Return the shape of the data."""
        # This may be patched by pandas to support pseudo-2D operations.
        return (self.size,)

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the underlying data."""
        return len(self.shape)

    def __len__(self) -> int:
        """
        Length of this array.

        Returns
        -------
        length : int
        """
        return self.shape[0]

    @property
    def base(self) -> Union[pa.Array, pa.ChunkedArray]:
        """Return base object of the underlying data."""
        return self.data

    def all(self, skipna: bool = False) -> Optional[bool]:
        """Compute whether all boolean values are True."""
        if pa.types.is_boolean(self.data.type):
            return all_op(self.data, skipna=skipna)
        else:
            raise TypeError("Can only execute all on boolean arrays")

    def any(self, skipna: bool = False, **kwargs) -> Optional[bool]:
        """Compute whether any boolean value is True."""
        if pa.types.is_boolean(self.data.type):
            return any_op(self.data, skipna=skipna)
        else:
            raise TypeError("Can only execute all on boolean arrays")

    def sum(self, skipna: bool = True):
        """Return the sum of the values."""
        return self._reduce("sum", skipna=skipna)

    def _reduce(self, name: str, skipna: bool = True, **kwargs):
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
        elif name == "sum" and self.dtype._is_numeric:
            return sum_op(self.data, skipna=skipna)
        elif name == "max" and self.dtype._is_numeric:
            return max_op(self.data, skipna=skipna)
        elif name == "min" and self.dtype._is_numeric:
            return min_op(self.data, skipna=skipna)
        elif name == "mean" and self.dtype._is_numeric:
            return sum_op(self.data, skipna=skipna) / len(self.data)
        elif name == "prod" and self.dtype._is_numeric:
            return prod_op(self.data, skipna=skipna)
        elif name == "std" and self.dtype._is_numeric:
            return std_op(self.data, skipna=skipna)
        elif name == "skew" and self.dtype._is_numeric:
            return skew_op(self.data, skipna=skipna)
        elif name == "kurt" and self.dtype._is_numeric:
            return kurt_op(self.data, skipna=skipna)
        elif name == "var" and self.dtype._is_numeric:
            return var_op(self.data, skipna=skipna)
        elif name == "median" and self.dtype._is_numeric:
            return median_op(self.data, skipna=skipna)

        raise TypeError(
            "cannot perform {name} with type {dtype}".format(
                name=name, dtype=self.dtype
            )
        )

    def _as_pandas_scalar(self, arrow_scalar: pa.Scalar):
        scalar = arrow_scalar.as_py()
        if scalar is None:
            return self._dtype.na_value
        else:
            return scalar

    def __array_ufunc__(self, ufunc, method: str, *inputs, **kwargs):
        """Apply a NumPy ufunc on the ExtensionArray."""
        if method != "__call__":
            if (
                method == "reduce"
                and getattr(ufunc, "__name__") == "logical_or"
                and self.dtype.arrow_dtype.id == 1
            ):
                return any_op(self.data, skipna=False)
            else:
                raise NotImplementedError(
                    f"Only method == '__call__' is supported in ufuncs, not '{method}'"
                )
        if len(inputs) == 1:
            if getattr(ufunc, "__name__") == "isnan":
                return self.isna()
            else:
                raise NotImplementedError(
                    f"ufunc with single input not supported: {ufunc}"
                )
        if len(inputs) != 2:
            raise NotImplementedError("Only ufuncs with a second input are supported")
        if len(kwargs) > 0:
            raise NotImplementedError("ufuncs with kwargs aren't supported")
        if isinstance(inputs[0], FletcherBaseArray):
            left = inputs[0].data
        else:
            left = inputs[0]
        if isinstance(inputs[1], FletcherBaseArray):
            right = inputs[1].data
        else:
            right = inputs[1]
        return type(self)(np_ufunc_op(left, right, ufunc))

    def _np_ufunc_op(self, op: Callable, other):
        """Apply a NumPy ufunc on the instance and any other object."""
        if isinstance(other, (pd.Series, pd.DataFrame)):
            return NotImplemented
        if isinstance(other, FletcherBaseArray):
            other = other.data
        return type(self)(np_ufunc_op(self.data, other, op))

    def _np_compare_op(self, op: Callable, np_op: Callable, other):
        """Apply a NumPy-based comparison on the instance and any other object."""
        if isinstance(other, (pd.Series, pd.DataFrame)):
            return NotImplemented
        # TODO: Only numeric comparisons are fast currently
        if not self.dtype._is_numeric:
            if isinstance(other, FletcherBaseArray):
                other = other.data.to_pandas()
            return type(self)(op(self.data.to_pandas(), other))
        return self._np_ufunc_op(np_op, other)

    __eq__ = partialmethod(  # type: ignore
        _np_compare_op, operator.eq, np.ndarray.__eq__
    )
    __ne__ = partialmethod(  # type: ignore
        _np_compare_op, operator.ne, np.ndarray.__ne__
    )
    __le__ = partialmethod(_np_compare_op, operator.le, np.ndarray.__le__)
    __lt__ = partialmethod(_np_compare_op, operator.lt, np.ndarray.__lt__)
    __ge__ = partialmethod(_np_compare_op, operator.ge, np.ndarray.__ge__)
    __gt__ = partialmethod(_np_compare_op, operator.gt, np.ndarray.__gt__)

    __add__ = partialmethod(_np_ufunc_op, np.ndarray.__add__)
    __radd__ = partialmethod(_np_ufunc_op, np.ndarray.__radd__)
    __sub__ = partialmethod(_np_ufunc_op, np.ndarray.__sub__)
    __rsub__ = partialmethod(_np_ufunc_op, np.ndarray.__rsub__)
    __mul__ = partialmethod(_np_ufunc_op, np.ndarray.__mul__)
    __rmul__ = partialmethod(_np_ufunc_op, np.ndarray.__rmul__)
    __floordiv__ = partialmethod(_np_ufunc_op, np.ndarray.__floordiv__)
    __rfloordiv__ = partialmethod(_np_ufunc_op, np.ndarray.__rfloordiv__)
    __truediv__ = partialmethod(_np_ufunc_op, np.ndarray.__truediv__)
    __rtruediv__ = partialmethod(_np_ufunc_op, np.ndarray.__rtruediv__)
    __pow__ = partialmethod(_np_ufunc_op, np.ndarray.__pow__)
    __rpow__ = partialmethod(_np_ufunc_op, np.ndarray.__rpow__)
    __mod__ = partialmethod(_np_ufunc_op, np.ndarray.__mod__)
    __rmod__ = partialmethod(_np_ufunc_op, np.ndarray.__rmod__)

    def __or__(self, other):
        """Compute vectorised or."""
        if not pa.types.is_boolean(self.dtype.arrow_dtype):
            raise NotImplementedError("__or__ is only supported for boolean arrays yet")

        if other is pd.NA or (pd.api.types.is_scalar(other) and pd.isna(other)):
            # All fields that are True stay True, all others get set to NA
            return type(self)(or_na(self.data))
        elif isinstance(other, bool):
            if other:
                # or with True yields all-True
                return type(self)(all_true(self.data))
            else:
                return self
        else:
            if isinstance(other, FletcherBaseArray):
                other = other.data
            return type(self)(or_vectorised(self.data, other))

    def __divmod__(self, other):
        """Compute divmod via floordiv and mod."""
        return (self.__floordiv__(other), self.__mod__(other))

    def unique(self):
        """
        Compute the ExtensionArray of unique values.

        It relies on the Pyarrow.ChunkedArray.unique and if
        it fails, comes back to the naive implementation.

        Returns
        -------
        uniques : ExtensionArray
        """
        try:
            return type(self)(self.data.unique())
        except NotImplementedError:
            return super().unique()

    def _pd_object_take(
        self,
        indices: Union[Sequence[int], np.ndarray],
        allow_fill: bool = False,
        fill_value: Optional[Any] = None,
    ) -> ExtensionArray:
        """Run take using object dtype and pandas' built-in algorithm.

        This is slow and should be avoided in future but is kept here as not all
        special cases are yet supported.
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

    def _take_array(
        self,
        array: pa.Array,
        indices: Union[Sequence[int], np.ndarray],
        allow_fill: bool = False,
        fill_value: Optional[Any] = None,
    ) -> ExtensionArray:
        """
        Take elements from a pyarrow.Array.

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
        if isinstance(indices, pa.Array) and pa.types.is_integer(indices):
            # TODO: handle allow_fill, fill_value
            if allow_fill or fill_value is not None:
                raise NotImplementedError(
                    "Cannot use allow_fill or fill_value with a pa.Array"
                )
            indices_array = indices
        elif isinstance(indices, Iterable):
            # Why is np.ndarray inferred as Iterable[Any]?
            if len(indices) == 0:  # type: ignore
                return type(self)(pa.array([], type=array.type))
            elif not is_array_like(indices):
                indices = np.array(indices)
            if not is_integer_dtype(indices):
                raise ValueError("Only integer dtyped indices are supported")
            # TODO: handle fill_value
            mask = indices < 0
            if allow_fill and indices.min() < -1:
                raise ValueError(
                    "Invalid value in 'indices'. Must be between -1 "
                    "and the length of the array."
                )
            if len(self) == 0 and (~mask).any():
                raise IndexError("cannot do a non-empty take")
            if indices.max() >= len(self):
                raise IndexError("out of bounds value in 'indices'.")
            if not allow_fill:
                indices[mask] = len(array) + indices[mask]
                mask = None
            elif not pd.isna(fill_value):
                # TODO: Needs fillna on pa.Array
                return self._pd_object_take(
                    indices, allow_fill=True, fill_value=fill_value
                )
            indices_array = pa.array(indices, mask=mask)
        elif is_array_like(indices) and len(indices) == 0:
            indices_array = pa.array([], type=pa.int64())
        elif (
            self.dtype.is_list
            and self.data.flatten().null_count == 0
            and self.data.null_count == 0
            and self.data.flatten().dtype._is_numeric
        ):
            return take_on_pyarrow_list(self.data, indices)
        else:
            raise NotImplementedError(f"take is not implemented for {type(indices)}")
        return type(self)(array.take(indices_array))

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
            if copy:
                return copycopy(self)
            else:
                return self

        arrow_type = None
        arrow_class = None
        pandas_type = None
        if isinstance(dtype, FletcherChunkedDtype):
            arrow_type = dtype.arrow_dtype
            dtype = dtype.arrow_dtype.to_pandas_dtype()
            if isinstance(self, FletcherChunkedArray):
                arrow_class = type(self)
            else:
                arrow_class = FletcherChunkedArray
        elif isinstance(dtype, FletcherContinuousDtype):
            arrow_type = dtype.arrow_dtype
            dtype = dtype.arrow_dtype.to_pandas_dtype()
            if isinstance(self, FletcherContinuousArray):
                arrow_class = type(self)
            else:
                arrow_class = FletcherContinuousArray
        elif isinstance(dtype, pa.DataType):
            arrow_type = dtype
            dtype = dtype.to_pandas_dtype()
            arrow_class = type(self)
        elif isinstance(dtype, pd.StringDtype):
            pandas_type = dtype
            dtype = np.dtype(str)
        else:
            dtype = np.dtype(dtype)

        # NumPy's conversion of list->unicode is differently from Python's
        # default. We want to have the default Python output, so force it here.
        if (self.dtype._is_list) and dtype.kind == "U":
            result = np.array([str(x) for x in self.data.to_pylist()])
            if pandas_type is not None:
                return pd.array(result, dtype=pandas_type)
            else:
                return result

        if arrow_type is not None and arrow_class is not None:
            return arrow_class(np.asarray(self).astype(dtype), dtype=arrow_type)
        else:
            result = np.asarray(self).astype(dtype)
            if pandas_type is not None:
                return pd.array(result, dtype=pandas_type)
            else:
                return result

    def value_counts(self, dropna: bool = True) -> "pd.Series":
        """
        Return a Series containing counts of each unique value.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of missing values.

        Returns
        -------
        counts : Series

        See Also
        --------
        Series.value_counts
        """
        vc = self.data.value_counts()

        # Index cannot hold ExtensionArrays yet
        index = pd.Index(type(self)(vc.field(0)).astype(object))
        # No missings, so we can adhere to the interface and return a numpy array.
        counts = np.array(vc.field(1))

        if dropna and self.data.null_count > 0:
            raise NotImplementedError("yo")

        return pd.Series(counts, index=index)

    def isna(self) -> np.ndarray:
        """
        Boolean NumPy array indicating if each value is missing.

        This should return a 1-D array the same length as 'self'.
        """
        if hasattr(self.data, "is_null"):
            return np.array(self.data.is_null())
        else:
            # Remove once drop support for pyarrow<1
            return extract_isnull_bytemap(self.data)


class FletcherContinuousArray(FletcherBaseArray):
    """Pandas ExtensionArray implementation backed by Apache Arrow's pyarrow.Array."""

    def __init__(self, array, dtype=None, copy: Optional[bool] = None):
        # Copy is not used at the moment. It's only affect will be when we
        # allow array to be a FletcherContinuousArray
        if is_array_like(array) or isinstance(array, list):
            self.data = pa.array(array, type=dtype, from_pandas=True)
        elif isinstance(array, pa.Array):
            # TODO: Assert dtype
            self.data = array
        elif isinstance(array, pa.ChunkedArray):
            # TODO: Assert dtype
            if array.num_chunks == 1:
                self.data = array.chunk(0)
            else:
                self.data = pa.concat_arrays(array.iterchunks())
        else:
            raise ValueError(
                "Unsupported type passed for {}: {}".format(
                    self.__class__.__name__, type(array)
                )
            )
        self._dtype = FletcherContinuousDtype(self.data.type)

    @classmethod
    def _concat_same_type(cls, to_concat):
        # type: (Sequence[ExtensionArray]) -> ExtensionArray
        """Concatenate multiple array.

        Parameters
        ----------
        to_concat : sequence of this type

        Returns
        -------
        ExtensionArray
        """
        return cls(pa.concat_arrays([array.data for array in to_concat]))

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
        if PANDAS_GE_0_26_0:
            key = check_array_indexer(self, key)

        if self.dtype._is_list:
            # TODO: We can probably implement this for the scalar case?
            # TODO: Implement a list accessor and then the three mentioned methods
            raise ValueError(
                "__setitem__ is not supported for list types "
                "due to the ambiguity of the arguments, use .fr_list.setvalue, "
                ".fr_list.setslice or fr_list.setmask instead."
            )
        # Convert all possible input key types to an array of integers
        if is_bool_dtype(key):
            key_array = np.argwhere(key).flatten()
        elif isinstance(key, slice):
            key_array = np.array(range(len(self))[key])
        elif is_integer(key):
            key_array = np.array([key])
        else:
            key_array = np.asanyarray(key)

        if pd.api.types.is_scalar(value):
            if value is pd.NA:
                value = None
            value = np.broadcast_to(value, len(key_array))
        else:
            value = np.asarray(value)

        if len(key_array) != len(value):
            raise ValueError("Length mismatch between index and value.")

        arr = self.data.to_pandas().values
        # In the case where we zero-copy Arrow to Pandas conversion, the
        # the resulting arrays are read-only.
        if not arr.flags.writeable:
            arr = arr.copy()
        arr[key_array] = value

        mask = None
        # ARROW-2806: Inconsistent handling of np.nan requires adding a mask
        if (
            pa.types.is_integer(self.dtype.arrow_dtype)
            or pa.types.is_date(self.dtype.arrow_dtype)
            or pa.types.is_floating(self.dtype.arrow_dtype)
            or pa.types.is_boolean(self.dtype.arrow_dtype)
        ):
            nan_values = pd.isna(value)
            if any(nan_values):
                nan_index = key_array[nan_values]
                mask = np.zeros_like(arr, dtype=bool)
                mask[nan_index] = True
        self.data = pa.array(arr, self.dtype.arrow_dtype, mask=mask)

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
        if PANDAS_GE_0_26_0:
            item = check_array_indexer(self, item)

        # Arrow 0.18+ supports slices perfectly
        if isinstance(item, slice) and not ARROW_GE_0_18_0:
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
            if is_integer_dtype(item) or len(item) == 0:
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
            item = int(item)
            if item < 0:
                item += len(self)
            if item >= len(self):
                return None
        value = self.data[item]
        if isinstance(value, pa.Array):
            return type(self)(value)
        else:
            return self._as_pandas_scalar(value)

    def copy(self):
        # type: () -> ExtensionArray
        """
        Return a copy of the array.

        Currently is a shadow copy - pyarrow array are supposed to be immutable.

        Returns
        -------
        ExtensionArray
        """
        return type(self)(self.data)

    @property
    def nbytes(self):
        # type: () -> int
        """Return the number of bytes needed to store this object in memory."""
        size = 0
        for buf in self.data.buffers():
            if buf is not None:
                size += buf.size
        return size

    @doc(ExtensionArray.factorize)
    def factorize(self, na_sentinel=-1):
        if pa.types.is_dictionary(self.data.type):
            indices = self.data.indices.to_pandas()
            return indices.values, type(self)(self.data.dictionary)
        else:
            # Dictionaryencode and do the same as above
            encoded = self.data.dictionary_encode()
            indices = encoded.indices.to_pandas()
            if indices.dtype.kind == "f":
                indices[np.isnan(indices)] = na_sentinel
                indices = indices.astype(np.int64)
            if not is_int64_dtype(indices):
                indices = indices.astype(np.int64)
            return indices.values, type(self)(encoded.dictionary)

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
        if isinstance(scalars, FletcherContinuousArray):
            return scalars
        if not ARROW_GE_0_18_0:
            scalars = [None if x is pd.NA else x for x in scalars]
        if dtype and isinstance(dtype, FletcherContinuousDtype):
            dtype = dtype.arrow_dtype
        return cls(pa.array(scalars, type=dtype, from_pandas=True))

    def fillna(self, value=None, method=None, limit=None):
        """Fill NA/NaN values using the specified method.

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
        import pandas.core.missing as pd_missing

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
                # pandas 1.2+ doesn't expose pad_1d anymore
                if not hasattr(pd_missing, "pad_1d"):
                    func = pd_missing.get_fill_func(method)
                else:
                    func = (
                        pd_missing.pad_1d if method == "pad" else pd_missing.backfill_1d
                    )
                new_values = func(self.astype(object), limit=limit, mask=mask)
                new_values = self._from_sequence(new_values, self._dtype.arrow_dtype)
            else:
                # fill with value
                new_values = self.copy()
                new_values[mask] = value
        else:
            new_values = self.copy()
        return new_values

    def take(
        self,
        indices: Union[Sequence[int], np.ndarray],
        allow_fill: bool = False,
        fill_value: Optional[Any] = None,
    ) -> ExtensionArray:
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
        return self._take_array(self.data, indices, allow_fill, fill_value)

    def flatten(self):
        """
        Flatten the array.
        """
        return type(self)(self.data.flatten())


class FletcherChunkedArray(FletcherBaseArray):
    """Pandas ExtensionArray implementation backed by Apache Arrow."""

    _can_hold_na = True

    def __init__(self, array, dtype=None, copy=None):
        # Copy is not used at the moment. It's only affect will be when we
        # allow array to be a FletcherChunkedArray
        if is_array_like(array) or isinstance(array, list):
            self.data = pa.chunked_array(
                [pa.array(array, type=dtype, from_pandas=True)]
            )
        elif isinstance(array, pa.Array):
            # ARROW-7008: pyarrow.chunked_array([array]) fails on array with all-None buffers
            if len(array) == 0 and all(b is None for b in array.buffers()):
                array = pa.array([], type=array.type)
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
        self._dtype = FletcherChunkedDtype(self.data.type)
        self.offsets = self._calculate_chunk_offsets()

    @classmethod
    def _concat_same_type(cls, to_concat):
        # type: (Sequence[ExtensionArray]) -> ExtensionArray
        """Concatenate multiple array.

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

    def _calculate_chunk_offsets(self) -> np.ndarray:
        """Return an array holding the indices pointing to the first element of each chunk."""
        return _calculate_chunk_offsets(self.data)

    def _get_chunk_indexer(self, array):
        """Return an array with the chunk number for each index."""
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
        if PANDAS_GE_0_26_0:
            key = check_array_indexer(self, key)

        if self.dtype._is_list:
            # TODO: We can probably implement this for the scalar case?
            # TODO: Implement a list accessor and then the three mentioned methods
            raise ValueError(
                "__setitem__ is not supported for list types "
                "due to the ambiguity of the arguments, use .fr_list.setvalue, "
                ".fr_list.setslice or fr_list.setmask instead."
            )
        # Convert all possible input key types to an array of integers
        if is_bool_dtype(key):
            key_array = np.argwhere(key).flatten()
        elif isinstance(key, slice):
            key_array = np.array(range(len(self))[key])
        elif is_integer(key):
            key_array = np.array([key])
        else:
            key_array = np.asanyarray(key)

        if pd.api.types.is_scalar(value):
            value = np.broadcast_to(value, len(key_array))
        else:
            value = np.asarray(value)

        if len(key_array) != len(value):
            raise ValueError("Length mismatch between index and value.")

        affected_chunks_index = self._get_chunk_indexer(key_array)
        affected_chunks_unique = np.unique(affected_chunks_index)

        all_chunks = list(self.data.iterchunks())

        for ix, offset in zip(
            affected_chunks_unique, self.offsets[affected_chunks_unique]
        ):
            chunk = all_chunks[ix]

            # Translate the array-wide indices to indices of the chunk
            key_chunk_indices = np.argwhere(affected_chunks_index == ix).flatten()
            array_chunk_indices = key_array[key_chunk_indices] - offset

            arr = chunk.to_pandas().values
            # In the case where we zero-copy Arrow to Pandas conversion, the
            # the resulting arrays are read-only.
            if not arr.flags.writeable:
                arr = arr.copy()
            arr[array_chunk_indices] = value[key_chunk_indices]

            mask = None
            # ARROW-2806: Inconsistent handling of np.nan requires adding a mask
            if (
                pa.types.is_integer(self.dtype.arrow_dtype)
                or pa.types.is_date(self.dtype.arrow_dtype)
                or pa.types.is_floating(self.dtype.arrow_dtype)
                or pa.types.is_boolean(self.dtype.arrow_dtype)
            ):
                nan_values = pd.isna(value[key_chunk_indices])
                if any(nan_values):
                    nan_index = array_chunk_indices[nan_values]
                    mask = np.zeros_like(arr, dtype=bool)
                    mask[nan_index] = True
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
        if PANDAS_GE_0_26_0:
            item = check_array_indexer(self, item)

        # Arrow 0.18+ supports slices perfectly
        if isinstance(item, slice) and not ARROW_GE_0_18_0:
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
            if is_integer_dtype(item) or (len(item) == 0):
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
            item = int(item)
            if item < 0:
                item += len(self)
            if item >= len(self):
                return None
        value = self.data[item]
        if isinstance(value, pa.ChunkedArray):
            return type(self)(value)
        else:
            return self._as_pandas_scalar(value)

    def copy(self):
        # type: () -> ExtensionArray
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
        return type(self)(self.data)

    @property
    def nbytes(self):
        # type: () -> int
        """Return the number of bytes needed to store this object in memory."""
        size = 0
        for chunk in self.data.chunks:
            for buf in chunk.buffers():
                if buf is not None:
                    size += buf.size
        return size

    @doc(ExtensionArray.factorize)
    def factorize(self, na_sentinel=-1):
        if pa.types.is_dictionary(self.data.type):
            raise NotImplementedError()
        else:
            # Dictionaryencode and do the same as above
            encoded = self.data.dictionary_encode()
            indices = pa.chunked_array(
                [c.indices for c in encoded.chunks], type=encoded.type.index_type
            ).to_pandas()
            if indices.dtype.kind == "f":
                indices[np.isnan(indices)] = na_sentinel
                indices = indices.astype(int)
            if not is_int64_dtype(indices):
                indices = indices.astype(np.int64)
            if encoded.num_chunks == 0:
                return (
                    indices.values,
                    type(self)(pa.array([], type=encoded.type.value_type)),
                )
            else:
                return indices.values, type(self)(encoded.chunk(0).dictionary)

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
        if isinstance(scalars, FletcherChunkedArray):
            return scalars
        if not ARROW_GE_0_18_0:
            scalars = [None if x is pd.NA else x for x in scalars]
        if dtype and isinstance(dtype, FletcherChunkedDtype):
            dtype = dtype.arrow_dtype
        return cls(pa.array(scalars, type=dtype, from_pandas=True))

    def fillna(self, value=None, method=None, limit=None):
        """Fill NA/NaN values using the specified method.

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
        import pandas.core.missing as pd_missing

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
                # pandas 1.2+ doesn't expose pad_1d anymore
                if not hasattr(pd_missing, "pad_1d"):
                    func = pd_missing.get_fill_func(method)
                else:
                    func = (
                        pd_missing.pad_1d if method == "pad" else pd_missing.backfill_1d
                    )
                new_values = func(self.astype(object), limit=limit, mask=mask)
                new_values = self._from_sequence(new_values, self._dtype.arrow_dtype)
            else:
                # fill with value
                new_values = self.copy()
                new_values[mask] = value
        else:
            new_values = self.copy()
        return new_values

    def take(
        self,
        indices: Union[Sequence[int], np.ndarray],
        allow_fill: bool = False,
        fill_value: Optional[Any] = None,
    ) -> ExtensionArray:
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
        if self.data.num_chunks == 1:
            return self._take_array(self.data.chunk(0), indices, allow_fill, fill_value)

        from pandas.core.algorithms import take

        data = self.astype(object)
        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value
        # fill value should always be translated from the scalar
        # type for the array, to the physical storage type for
        # the data, before passing to take.
        result = take(data, indices, fill_value=fill_value, allow_fill=allow_fill)
        return self._from_sequence(result, dtype=self.data.type)

    def flatten(self):
        """
        Flatten the array.
        """
        return type(self)(
            pa.chunked_array(ch.flatten() for ch in self.data.iterchunks())
        )


def pandas_from_arrow(
    arrow_object: Union[pa.RecordBatch, pa.Table, pa.Array, pa.ChunkedArray],
    continuous: bool = False,
):
    """
    Convert Arrow object instance to their Pandas equivalent by using Fletcher.

    The conversion rules are:
      * {RecordBatch, Table} -> DataFrame
      * {Array, ChunkedArray} -> Series

    Parameters
    ----------
    arrow_object : RecordBatch, Table, Array or ChunkedArray
        object to be converted
    continuous : bool
        Use FletcherContinuousArray instead of FletcherChunkedArray
    """
    if continuous:
        array_type = FletcherContinuousArray
    else:
        array_type = FletcherChunkedArray
    if isinstance(arrow_object, pa.RecordBatch):
        data: OrderedDict = OrderedDict()
        for ix, arr in enumerate(arrow_object):
            col_name = arrow_object.schema.names[ix]
            data[col_name] = array_type(arr)
        return pd.DataFrame(data)
    elif isinstance(arrow_object, pa.Table):
        data = OrderedDict()
        for name, col in zip(arrow_object.column_names, arrow_object.itercolumns()):
            data[name] = array_type(col)
        return pd.DataFrame(data)
    elif isinstance(arrow_object, (pa.ChunkedArray, pa.Array)):
        return pd.Series(array_type(arrow_object))
    else:
        raise NotImplementedError(
            "Objects of type {} are not supported".format(type(arrow_object))
        )


__all__: List[str] = []
