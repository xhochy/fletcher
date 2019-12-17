from functools import partial, singledispatch
from typing import Any, Callable, List, Optional, Tuple, Union

import numba
import numpy as np
import pyarrow as pa
from pandas.core import nanops

from ._numba_compat import NumbaStringArray


def _calculate_chunk_offsets(chunked_array: pa.ChunkedArray) -> np.ndarray:
    """Return an array holding the indices pointing to the first element of each chunk."""
    offset = 0
    offsets = []
    for chunk in chunked_array.iterchunks():
        offsets.append(offset)
        offset += len(chunk)
    return np.array(offsets)


def _extract_data_buffer_as_np_array(array: pa.Array) -> np.ndarray:
    """Extract the data buffer of a numeric-typed pyarrow.Array as an np.ndarray."""
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


def _in_chunk_offsets(
    arr: pa.ChunkedArray, offsets: List[int]
) -> List[Tuple[int, int, int]]:
    new_offsets = []
    pos = 0
    chunk = 0
    chunk_pos = 0
    for offset, offset_next in zip(offsets, offsets[1:] + [len(arr)]):
        diff = offset - pos
        chunk_remains = len(arr.chunk(chunk)) - chunk_pos
        step = offset_next - offset
        if diff == chunk_remains:
            chunk += 1
            chunk_pos = 0
            pos += chunk_remains
            new_offsets.append((chunk, chunk_pos, step))
        elif diff == 0:  # The first offset
            new_offsets.append((chunk, chunk_pos, step))
        else:  # diff < chunk_remains
            chunk_pos += diff
            pos += diff
            new_offsets.append((chunk, chunk_pos, step))
    return new_offsets


def _combined_in_chunk_offsets(
    a: pa.ChunkedArray, b: pa.ChunkedArray
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    offsets_a = _calculate_chunk_offsets(a)
    offsets_b = _calculate_chunk_offsets(b)
    offsets = sorted(set(list(offsets_a) + list(offsets_b)))
    in_a_offsets = _in_chunk_offsets(a, offsets)
    in_b_offsets = _in_chunk_offsets(b, offsets)
    return in_a_offsets, in_b_offsets


@singledispatch
def np_ufunc_op(a: Any, b: Any, op: Callable):
    """Apply a NumPy ufunc where at least one of the arguments is an Arrow structure."""
    # a is neither a pa.Array nor a pa.ChunkedArray, we expect only numpy.ndarray or scalars.
    if isinstance(b, pa.ChunkedArray):
        if np.isscalar(a):
            new_chunks = []
            for chunk in b.iterchunks():
                new_chunks.append(np_ufunc_op(a, chunk, op))
            return pa.chunked_array(new_chunks)
        else:
            new_chunks = []
            offsets = _calculate_chunk_offsets(b)
            for chunk, offset in zip(b.iterchunks(), offsets):
                new_chunks.append(
                    np_ufunc_op(a[offset : offset + len(chunk)], chunk, op)
                )
            return pa.chunked_array(new_chunks)
    elif isinstance(b, pa.Array):
        # a is non-masked, either array-like or scalar
        # numpy can handle all types of b from here
        np_arr = _extract_data_buffer_as_np_array(b)
        mask = extract_isnull_bytemap(b)
        if np.isscalar(a):
            a = np.array(a)
        new_arr = op(a, np_arr)
        # Don't set type as we might have valid casts like int->float in truediv
        return pa.array(new_arr, mask=mask)
    else:
        # Should never be reached, add a safe-guard
        raise NotImplementedError(f"Cannot apply ufunc on {type(a)} and {type(b)}")


@np_ufunc_op.register
def _1(a: pa.ChunkedArray, b: Any, op: Callable):
    """Apply a NumPy ufunc where at least one of the arguments is an Arrow structure."""
    if isinstance(b, pa.ChunkedArray):
        in_a_offsets, in_b_offsets = _combined_in_chunk_offsets(a, b)

        new_chunks: List[pa.Array] = []
        for a_offset, b_offset in zip(in_a_offsets, in_b_offsets):
            a_slice = a.chunk(a_offset[0])[a_offset[1] : a_offset[1] + a_offset[2]]
            b_slice = b.chunk(b_offset[0])[b_offset[1] : b_offset[1] + b_offset[2]]
            new_chunks.append(np_ufunc_op(a_slice, b_slice, op))
        return pa.chunked_array(new_chunks)
    elif np.isscalar(b):
        new_chunks = []
        for chunk in a.iterchunks():
            new_chunks.append(np_ufunc_op(chunk, b, op))
        return pa.chunked_array(new_chunks)
    else:
        new_chunks = []
        offsets = _calculate_chunk_offsets(a)
        for chunk, offset in zip(a.iterchunks(), offsets):
            new_chunks.append(np_ufunc_op(chunk, b[offset : offset + len(chunk)], op))
        return pa.chunked_array(new_chunks)


@np_ufunc_op.register
def _2(a: pa.Array, b: Any, op: Callable):
    """Apply a NumPy ufunc where at least one of the arguments is an Arrow structure."""
    if isinstance(b, pa.ChunkedArray):
        new_chunks = []
        offsets = _calculate_chunk_offsets(b)
        for chunk, offset in zip(b.iterchunks(), offsets):
            new_chunks.append(np_ufunc_op(a[offset : offset + len(chunk)], chunk, op))
        return pa.chunked_array(new_chunks)
    elif isinstance(b, pa.Array):
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
    else:
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
