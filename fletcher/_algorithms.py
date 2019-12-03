from functools import partial
from typing import Callable, Union

import numba
import numpy as np
import pyarrow as pa
from pandas.core import nanops

from ._numba_compat import NumbaStringArray


def _extract_data_buffer_as_np_array(array: pa.Array) -> np.ndarray:
    # TODO: Write an explicit test for this with offsets
    dtype = array.type.to_pandas_dtype()
    start = array.offset
    end = array.offset + len(array)
    return np.asanyarray(array.buffers()[1]).view(dtype)[start:end]


@numba.jit(nogil=True, nopython=True)
def _extract_isnull_bytemap(
    bitmap: bytes,
    bitmap_length: int,
    bitmap_offset: int,
    dst_offset: int,
    dst: np.ndarray,
) -> None:
    """(internal) write the values of a valid bitmap as bytes to a pre-allocatored isnull bytemap.

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


@numba.jit(nogil=True, nopython=True)
def isnull(sa):
    result = np.empty(sa.size, np.uint8)
    _isnull(sa, 0, result)
    return result


@numba.jit(nogil=True, nopython=True)
def _isnull(sa, offset, out):
    for i in range(sa.size):
        out[offset + i] = sa.isnull(i)


@numba.jit(nogil=True, nopython=True)
def _startswith(sa, needle, na, offset, out):
    for i in range(sa.size):
        if sa.isnull(i):
            out[offset + i] = na
            continue

        if sa.byte_length(i) < needle.length:
            out[offset + i] = 0
            continue

        for j in range(needle.length):
            if sa.get_byte(i, j) != needle.get_byte(j):
                out[offset + i] = 0
                break

        else:
            out[offset + i] = 1


@numba.jit(nogil=True, nopython=True)
def _endswith(sa, needle, na, offset, out):
    for i in range(sa.size):
        if sa.isnull(i):
            out[offset + i] = na
            continue

        string_length = sa.byte_length(i)
        needle_length = needle.length
        if string_length < needle.length:
            out[offset + i] = 0
            continue

        for j in range(needle_length):
            if sa.get_byte(i, string_length - needle_length + j) != needle.get_byte(j):
                out[offset + i] = 0
                break

        else:
            out[offset + i] = 1


@numba.jit(nogil=True, nopython=True)
def str_length(sa):
    result = np.empty(sa.size, np.uint32)

    for i in range(sa.size):
        result[i] = sa.length(i)

    return result


@numba.jit(nogil=True, nopython=True)
def str_concat(sa1, sa2):
    # TODO: check overflow of size
    assert sa1.size == sa2.size

    result_missing = sa1.missing | sa2.missing
    result_offsets = np.zeros(sa1.size + 1, np.uint32)
    result_data = np.zeros(sa1.byte_size + sa2.byte_size, np.uint8)

    offset = 0
    for i in range(sa1.size):
        if sa1.isnull(i) or sa2.isnull(i):
            result_offsets[i + 1] = offset
            continue

        for j in range(sa1.byte_length(i)):
            result_data[offset] = sa1.get_byte(i, j)
            offset += 1

        for j in range(sa2.byte_length(i)):
            result_data[offset] = sa2.get_byte(i, j)
            offset += 1

        result_offsets[i + 1] = offset

    result_data = result_data[:offset]

    return NumbaStringArray(result_missing, result_offsets, result_data, 0)


@numba.njit(locals={"valid": numba.bool_, "value": numba.bool_})
def _any_op(length: int, valid_bits: bytes, data: bytes) -> int:
    for i in range(length):
        byte_offset = i // 8
        bit_offset = i % 8
        mask = np.uint8(1 << bit_offset)
        valid = valid_bits[byte_offset] & mask
        value = data[byte_offset] & mask
        if (valid and value) or (not valid):
            return True

    return False


@numba.njit(locals={"valid": numba.bool_, "value": numba.bool_})
def _any_op_skipna(length: int, valid_bits: bytes, data: bytes) -> bool:
    for i in range(length):
        byte_offset = i // 8
        bit_offset = i % 8
        mask = np.uint8(1 << bit_offset)
        valid = valid_bits[byte_offset] & mask
        value = data[byte_offset] & mask
        if valid and value:
            return True

    return False


@numba.njit(locals={"value": numba.bool_})
def _any_op_nonnull(length: int, data: bytes) -> bool:
    for i in range(length):
        byte_offset = i // 8
        bit_offset = i % 8
        mask = np.uint8(1 << bit_offset)
        value = data[byte_offset] & mask
        if value:
            return True

    return False


def any_op(arr: Union[pa.ChunkedArray, pa.Array], skipna: bool) -> bool:
    if isinstance(arr, pa.ChunkedArray):
        return any(any_op(chunk, skipna) for chunk in arr.chunks)

    if arr.null_count == 0:
        return _any_op_nonnull(len(arr), arr.buffers()[1])
    if skipna:
        return _any_op_skipna(len(arr), *arr.buffers())
    return _any_op(len(arr), *arr.buffers())


@numba.njit(locals={"valid": numba.bool_, "value": numba.bool_})
def _all_op(length: int, valid_bits: bytes, data: bytes) -> bool:
    # This may be specific to Pandas but we return True as long as there is not False in the data.
    for i in range(length):
        byte_offset = i // 8
        bit_offset = i % 8
        mask = np.uint8(1 << bit_offset)
        valid = valid_bits[byte_offset] & mask
        value = data[byte_offset] & mask
        if valid and not value:
            return False
    return True


@numba.njit(locals={"value": numba.bool_})
def _all_op_nonnull(length: int, data: bytes) -> bool:
    for i in range(length):
        byte_offset = i // 8
        bit_offset = i % 8
        mask = np.uint8(1 << bit_offset)
        value = data[byte_offset] & mask
        if not value:
            return False
    return True


def all_op(arr: Union[pa.ChunkedArray, pa.Array], skipna: bool) -> bool:
    if isinstance(arr, pa.ChunkedArray):
        return all(all_op(chunk, skipna) for chunk in arr.chunks)

    if arr.null_count == 0:
        return _all_op_nonnull(len(arr), arr.buffers()[1])
    # skipna is not relevant in the Pandas behaviour
    return _all_op(len(arr), *arr.buffers())


def np_op(npop: Callable, arr: Union[pa.ChunkedArray, pa.Array]):
    """Use numpy operations to provide a reduction."""
    if isinstance(arr, pa.ChunkedArray):
        return npop([np_op(npop, chunk) for chunk in arr.iterchunks()])
    else:
        np_arr = _extract_data_buffer_as_np_array(arr)
        mask = extract_isnull_bytemap(arr)
        return npop(np_arr[~mask])


sum_op = partial(np_op, np.sum)
max_op = partial(np_op, np.amax)
min_op = partial(np_op, np.amin)
prod_op = partial(np_op, np.prod)


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
