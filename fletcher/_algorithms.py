from functools import partial
from typing import Any, Callable, Optional, Union

import numpy as np
import pyarrow as pa
from numba import prange
from pandas.core import nanops

from fletcher._compat import njit
from fletcher.algorithms.utils.chunking import dispatch_chunked_binary_map


def _extract_data_buffer_as_np_array(array: pa.Array) -> np.ndarray:
    """Extract the data buffer of a numeric-typed pyarrow.Array as an np.ndarray."""
    dtype = array.type.to_pandas_dtype()
    start = array.offset
    end = array.offset + len(array)
    if pa.types.is_boolean(array.type):
        return np.unpackbits(
            _buffer_to_view(array.buffers()[1]).view(np.uint8), bitorder="little"
        )[start:end].astype(bool)
    else:
        return _buffer_to_view(array.buffers()[1]).view(dtype)[start:end]


EMPTY_BUFFER_VIEW = np.array([], dtype=np.uint8)


def _buffer_to_view(buf: Optional[pa.Buffer]) -> np.ndarray:
    """Extract the pyarrow.Buffer as np.ndarray[np.uint8]."""
    if buf is None:
        return EMPTY_BUFFER_VIEW
    else:
        return np.asanyarray(buf).view(np.uint8)


@njit
def _extract_isnull_bytemap(
    bitmap: bytes,
    bitmap_length: int,
    bitmap_offset: int,
    dst_offset: int,
    dst: np.ndarray,
) -> None:
    """Write the values of a valid bitmap as bytes to a pre-allocatored isnull bytemap.

    Parameters
    ----------
    bitmap: pyarrow.Buffer
        bitmap where a set bit indicates that a value is valid
    bitmap_length: int
        Number of bits to read from the bitmap
    bitmap_offset: int
        Number of bits to skip from the beginning of the bitmap.
    dst_offset: int
        Number of bytes to skip from the beginning of the output
    dst: numpy.array(dtype=bool)
        Pre-allocated numpy array where a byte is set when a value is null
    """
    for i in range(bitmap_length):
        idx = bitmap_offset + i
        byte_idx = idx // 8
        bit_mask = 1 << (idx % 8)
        dst[dst_offset + i] = (bitmap[byte_idx] & bit_mask) == 0


def extract_isnull_bytemap(array: Union[pa.ChunkedArray, pa.Array]) -> np.ndarray:
    """
    Extract the valid bitmaps of a (chunked) array into numpy isnull bytemaps.

    Parameters
    ----------
    array
        Array from which we extract the validity bits as bytes

    Returns
    -------
    valid_bytemap
    """
    if array.null_count == len(array):
        return np.ones(len(array), dtype=bool)

    if isinstance(array, pa.ChunkedArray):
        result = np.zeros(len(array), dtype=bool)
        if array.null_count == 0:
            return result

        offset = 0
        for chunk in array.chunks:
            if chunk.null_count > 0:
                _extract_isnull_bytemap(
                    chunk.buffers()[0], len(chunk), chunk.offset, offset, result
                )
            offset += len(chunk)
    else:
        valid_bitmap = array.buffers()[0]
        if valid_bitmap:
            # TODO: Can we use np.empty here to improve performance?
            result = np.zeros(len(array), dtype=bool)
            # TODO(ARROW-2664): We only need to following line to support
            #   executing the code in disabled-JIT mode.
            buf = memoryview(valid_bitmap)
            _extract_isnull_bytemap(buf, len(array), array.offset, 0, result)
        else:
            result = np.full(len(array), False)

    return result


@njit
def isnull(sa):
    result = np.empty(sa.size, np.uint8)
    _isnull(sa, 0, result)
    return result


@njit
def _isnull(sa, offset, out):
    for i in range(sa.size):
        out[offset + i] = sa.isnull(i)


@njit
def str_length(sa):
    result = np.empty(sa.size, np.uint32)

    for i in range(sa.size):
        result[i] = sa.length(i)

    return result


def np_reduce_op(
    npop: Callable,
    arr: Union[pa.ChunkedArray, pa.Array],
    skipna: bool = True,
    identity: Optional[int] = None,
):
    """Use numpy operations to provide a reduction."""
    if not skipna and arr.null_count > 0:
        return pa.NA
    elif isinstance(arr, pa.ChunkedArray):
        valid_chunks = [
            chunk for chunk in arr.iterchunks() if chunk.null_count != len(chunk)
        ]
        results = pa.array(
            [
                np_reduce_op(npop, chunk, skipna=skipna, identity=identity)
                for chunk in valid_chunks
            ]
        )
        return np_reduce_op(npop, results, skipna=skipna, identity=identity)
    else:
        if len(arr) == 0:
            if identity is None:
                raise ValueError("zero-size reduction on operation with no identity")
            else:
                return identity
        np_arr = _extract_data_buffer_as_np_array(arr)
        if arr.null_count > 0:
            mask = extract_isnull_bytemap(arr)
            return npop(np_arr[~mask])
        else:
            return npop(np_arr)


sum_op = partial(np_reduce_op, np.sum, identity=0)
max_op = partial(np_reduce_op, np.amax)
min_op = partial(np_reduce_op, np.amin)
prod_op = partial(np_reduce_op, np.prod, identity=1)


def pd_nanop(nanop: Callable, arr: Union[pa.ChunkedArray, pa.Array], skipna: bool):
    """Use pandas.core.nanops to provide a reduction."""
    if isinstance(arr, pa.ChunkedArray):
        data = pa.concat_arrays(arr.iterchunks())
    else:
        data = arr
    np_arr = _extract_data_buffer_as_np_array(data)
    mask = extract_isnull_bytemap(data)

    return nanop(np_arr, skipna=skipna, mask=mask)


std_op = partial(pd_nanop, nanops.nanstd)
skew_op = partial(pd_nanop, nanops.nanskew)
kurt_op = partial(pd_nanop, nanops.nankurt)
var_op = partial(pd_nanop, nanops.nanvar)
median_op = partial(pd_nanop, nanops.nanmedian)


def np_ufunc_array_array(a: pa.Array, b: pa.Array, op: Callable):
    np_arr_a = _extract_data_buffer_as_np_array(a)
    np_arr_b = _extract_data_buffer_as_np_array(b)
    if a.null_count > 0 and b.null_count > 0:
        # TODO: Combine them before extracting
        mask_a = extract_isnull_bytemap(a)
        mask_b = extract_isnull_bytemap(b)
        mask = mask_a | mask_b
    elif a.null_count > 0:
        mask = extract_isnull_bytemap(a)
    elif b.null_count > 0:
        mask = extract_isnull_bytemap(b)
    else:
        mask = None

    new_arr = op(np_arr_a, np_arr_b)
    # Don't set type as we might have valid casts like int->float in truediv
    return pa.array(new_arr, mask=mask)


def np_ufunc_array_scalar(a: pa.Array, b: Any, op: Callable):
    # b is non-masked, either array-like or scalar
    # numpy can handle all types of b from here
    np_arr = _extract_data_buffer_as_np_array(a)
    if a.null_count > 0:
        mask = extract_isnull_bytemap(a)
    else:
        mask = None
    new_arr = op(np_arr, b)
    # Don't set type as we might have valid casts like int->float in truediv
    return pa.array(new_arr, mask=mask)


def np_ufunc_scalar_array(a: Any, b: pa.Array, op: Callable):
    # a is non-masked, either array-like or scalar
    # numpy can handle all types of b from here
    np_arr = _extract_data_buffer_as_np_array(b)
    mask = extract_isnull_bytemap(b)
    if np.isscalar(a):
        a = np.array(a)
    new_arr = op(a, np_arr)
    # Don't set type as we might have valid casts like int->float in truediv
    return pa.array(new_arr, mask=mask)


def np_ufunc_op(a: Any, b: Any, op: Callable):
    ops = {
        "array_array": partial(np_ufunc_array_array, op=op),
        "array_nparray": partial(np_ufunc_array_scalar, op=op),
        "nparray_array": partial(np_ufunc_scalar_array, op=op),
        "array_scalar": partial(np_ufunc_array_scalar, op=op),
        "scalar_array": partial(np_ufunc_scalar_array, op=op),
    }
    return dispatch_chunked_binary_map(a, b, ops)


@njit
def _merge_non_aligned_bitmaps(
    valid_a: np.ndarray,
    inner_offset_a: int,
    valid_b: np.ndarray,
    inner_offset_b: int,
    length: int,
    result: np.ndarray,
) -> None:
    for i in range(length):
        a_pos = inner_offset_a + i
        byte_offset_a = a_pos // 8
        bit_offset_a = a_pos % 8
        mask_a = np.uint8(1 << bit_offset_a)
        value_a = valid_a[byte_offset_a] & mask_a

        b_pos = inner_offset_b + i
        byte_offset_b = b_pos // 8
        bit_offset_b = b_pos % 8
        mask_b = np.uint8(1 << bit_offset_b)
        value_b = valid_b[byte_offset_b] & mask_b

        byte_offset_result = i // 8
        bit_offset_result = i % 8
        mask_result = np.uint8(1 << bit_offset_result)

        current = result[byte_offset_result]
        if (
            value_a and value_b
        ):  # must be logical, not bit-wise as different bits may be flagged
            result[byte_offset_result] = current | mask_result
        else:
            result[byte_offset_result] = current & ~mask_result


def _extract_isnull_bitmap(arr: pa.Array, offset: int, length: int):
    """
    Extract isnull bitmap with offset and padding.

    Ensures that even when pyarrow does return an empty bitmap that a filled
    one will be returned.
    """
    buf = _buffer_to_view(arr.buffers()[0])
    if len(buf) > 0:
        return buf[offset : offset + length]
    else:
        return np.full(length, fill_value=255, dtype=np.uint8)


def _merge_valid_bitmaps(a: pa.Array, b: pa.Array) -> np.ndarray:
    """Merge two valid masks of pyarrow.Array instances.

    This method already assumes that both arrays are of the same length.
    This property is not checked again.
    """
    length = len(a) // 8
    if len(a) % 8 != 0:
        length += 1

    offset_a = a.offset // 8
    if a.offset % 8 != 0:
        pad_a = 1
    else:
        pad_a = 0
    valid_a = _extract_isnull_bitmap(a, offset_a, length + pad_a)

    offset_b = b.offset // 8
    if b.offset % 8 != 0:
        pad_b = 1
    else:
        pad_b = 0
    valid_b = _extract_isnull_bitmap(b, offset_b, length + pad_b)

    if a.offset % 8 == 0 and b.offset % 8 == 0:
        result = valid_a & valid_b

        # Mark trailing bits with 0
        if len(a) % 8 != 0:
            result[-1] = result[-1] & (2 ** (len(a) % 8) - 1)
        return result
    else:
        # Allocate result
        result = np.zeros(length, dtype=np.uint8)

        inner_offset_a = a.offset % 8
        inner_offset_b = b.offset % 8
        # TODO: We can optimite this when inner_offset_a == inner_offset_b
        _merge_non_aligned_bitmaps(
            valid_a, inner_offset_a, valid_b, inner_offset_b, len(a), result
        )

        return result


@njit(fastmath=True)
def _get_new_indptr(self_indptr, indices, new_indptr):
    for i in range(len(indices)):
        row = indices[i]
        new_indptr[i + 1] = new_indptr[i] + self_indptr[row + 1] - self_indptr[row]
    return new_indptr


@njit(fastmath=True, parallel=True)
def _fill_up_indices(new_indptr, new_indices, self_indices, self_indptr, indices):
    for i in prange(len(indices)):
        row = indices[i]
        size = self_indptr[row + 1] - self_indptr[row]
        if size > 0:
            new_indices[new_indptr[i] : new_indptr[i + 1]] = self_indices[
                self_indptr[row] : self_indptr[row + 1]
            ]


def take_on_pyarrow_list(array, indices):
    """Return a pyarrow.ListArray or pyarrow.LargeListArray containing only the rows of the given indices."""
    if len(array.flatten()) == 0:
        return array.take(pa.array(indices))

    self_indptr, self_indices = map(np.asarray, (array.offsets, array.values))
    self_indices = self_indices[self_indptr[0] : self_indptr[-1]]  # this is only a view
    self_indptr = self_indptr - self_indptr[0]  # this will make a copy !
    self_indptr.setflags(write=0)
    array.validate()

    length = indices.shape[0]

    # let's start with larger np.int64 dtype to avoid overflow
    # and yes, it will require a bit more memory ...
    new_indptr = np.zeros(length + 1, dtype=np.int64)
    new_indptr = _get_new_indptr(self_indptr, indices, new_indptr)
    # now we can see if we can downcast
    if new_indptr[-1] < np.iinfo(np.int32).max:
        new_indptr = new_indptr.astype(np.int32)
        pa_type = pa.ListArray
    else:
        pa_type = pa.LargeListArray

    new_indices = np.zeros(new_indptr[-1], dtype=self_indices.dtype)
    _fill_up_indices(new_indptr, new_indices, self_indices, self_indptr, indices)
    return pa_type.from_arrays(new_indptr, new_indices)
