from typing import Any, Union

import numba
import numpy as np
import pyarrow as pa

from fletcher._compat import njit
from fletcher.algorithms.utils.chunking import (
    apply_per_chunk,
    dispatch_chunked_binary_map,
)


@njit(locals={"valid": numba.bool_, "value": numba.bool_})
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


@njit(locals={"valid": numba.bool_, "value": numba.bool_})
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


@njit(locals={"value": numba.bool_})
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
    """Perform any() on a boolean Arrow structure."""
    if isinstance(arr, pa.ChunkedArray):
        return any(any_op(chunk, skipna) for chunk in arr.chunks)

    if arr.null_count == 0:
        return _any_op_nonnull(len(arr), arr.buffers()[1])
    if skipna:
        return _any_op_skipna(len(arr), *arr.buffers())
    return _any_op(len(arr), *arr.buffers())


@njit(locals={"valid": numba.bool_, "value": numba.bool_})
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


@njit(locals={"value": numba.bool_})
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
    """Perform all() on a boolean Arrow structure."""
    if isinstance(arr, pa.ChunkedArray):
        return all(all_op(chunk, skipna) for chunk in arr.chunks)

    if arr.null_count == 0:
        return _all_op_nonnull(len(arr), arr.buffers()[1])
    # skipna is not relevant in the Pandas behaviour
    return _all_op(len(arr), *arr.buffers())


@njit(locals={"value": numba.bool_})
def _or_na(
    length: int, offset: int, valid_bits: bytes, data: bytes, output: np.ndarray
):
    null_count = 0
    for i in range(length):
        byte_offset = (i + offset) // 8
        bit_offset = (i + offset) % 8
        mask = np.uint8(1 << bit_offset)

        valid = valid_bits[byte_offset] & mask
        value = data[byte_offset] & mask

        if valid and value:
            byte_offset = i // 8
            bit_offset = i % 8
            mask = np.uint8(1 << bit_offset)

            output[byte_offset] = output[byte_offset] | mask
        else:
            null_count += 1

    return null_count


@apply_per_chunk
def or_na(arr: pa.Array) -> pa.Array:
    """Apply ``array | NA`` with a boolean pyarrow.Array."""
    output_length = len(arr) // 8
    if len(arr) % 8 != 0:
        output_length += 1

    if arr.null_count == 0:
        return pa.Array.from_buffers(
            pa.bool_(),
            len(arr),
            [arr.buffers()[1], arr.buffers()[1]],
            null_count=-1,
            offset=arr.offset,
        )
    else:
        output = np.zeros(output_length, dtype=np.uint8)
        null_count = _or_na(
            len(arr), arr.offset, arr.buffers()[0], arr.buffers()[1], output
        )
        buf = pa.py_buffer(output)
        return pa.Array.from_buffers(pa.bool_(), len(arr), [buf, buf], null_count)


@apply_per_chunk
def all_true_like(arr: pa.Array) -> pa.Array:
    """Return a boolean array with all-True with the same size as the input and the same valid bitmap."""
    valid_buffer = arr.buffers()[0]
    if valid_buffer:
        valid_buffer = valid_buffer.slice(arr.offset // 8)

    output_offset = arr.offset % 8
    output_length = len(arr) + output_offset

    output_size = output_length // 8
    if output_length % 8 > 0:
        output_size += 1
    output = np.full(output_size, fill_value=255, dtype=np.uint8)

    return pa.Array.from_buffers(
        pa.bool_(),
        len(arr),
        [valid_buffer, pa.py_buffer(output)],
        arr.null_count,
        output_offset,
    )


@apply_per_chunk
def all_true(arr: pa.Array) -> pa.Array:
    """Return a boolean array with all-True, all-valid with the same size ."""
    output_length = len(arr) // 8
    if len(arr) % 8 != 0:
        output_length += 1

    buf = pa.py_buffer(np.full(output_length, 255, dtype=np.uint8))
    return pa.Array.from_buffers(pa.bool_(), len(arr), [buf, buf], 0)


@njit(locals={"value_a": numba.bool_, "value_b": numba.bool_})
def bitmap_or_unaligned(
    length: int, a: bytes, offset_a: int, b: bytes, offset_b: int, result: np.ndarray
) -> None:
    """Perform OR on two bitmaps without any alignments requirements."""
    for i in range(length):
        a_pos = offset_a + i
        byte_offset_a = a_pos // 8
        bit_offset_a = a_pos % 8
        mask_a = np.uint8(1 << bit_offset_a)
        value_a = a[byte_offset_a] & mask_a

        b_pos = offset_b + i
        byte_offset_b = b_pos // 8
        bit_offset_b = b_pos % 8
        mask_b = np.uint8(1 << bit_offset_b)
        value_b = b[byte_offset_b] & mask_b

        byte_offset_result = i // 8
        bit_offset_result = i % 8
        mask_result = np.uint8(1 << bit_offset_result)

        current = result[byte_offset_result]
        if (
            value_a or value_b
        ):  # must be logical, not bit-wise as different bits may be flagged
            result[byte_offset_result] = current | mask_result
        else:
            result[byte_offset_result] = current & ~mask_result


@njit
def masked_bitmap_or_unaligned(
    length: int,
    valid_bits_a: bytes,
    a: bytes,
    offset_a: int,
    valid_bits_b: bytes,
    b: bytes,
    offset_b: int,
    result: np.ndarray,
    valid_bits: np.ndarray,
) -> int:
    """Perform OR on two bitmaps with valid bitmasks without any alignment requirements."""
    null_count = 0
    for i in range(length):
        a_pos = offset_a + i
        byte_offset_a = a_pos // 8
        bit_offset_a = a_pos % 8
        mask_a = np.uint8(1 << bit_offset_a)
        valid_a = valid_bits_a[byte_offset_a] & mask_a
        value_a = a[byte_offset_a] & mask_a

        b_pos = offset_b + i
        byte_offset_b = b_pos // 8
        bit_offset_b = b_pos % 8
        mask_b = np.uint8(1 << bit_offset_b)
        valid_b = valid_bits_b[byte_offset_b] & mask_b
        value_b = b[byte_offset_b] & mask_b

        byte_offset_result = i // 8
        bit_offset_result = i % 8
        mask_result = np.uint8(1 << bit_offset_result)

        current = result[byte_offset_result]
        current_valid = valid_bits[byte_offset_result]
        if (valid_a and value_a) or (valid_b and value_b):
            result[byte_offset_result] = current | mask_result
            valid_bits[byte_offset_result] = current_valid | mask_result
        elif valid_a and valid_b:
            # a and b are False
            result[byte_offset_result] = current & ~mask_result
            valid_bits[byte_offset_result] = current_valid | mask_result
        else:
            # One is False and at least one is Null
            valid_bits[byte_offset_result] = current_valid & ~mask_result
            null_count += 1

    return null_count


def or_array_array(a: pa.Array, b: pa.Array) -> pa.Array:
    """Perform ``pyarrow.Array | pyarrow.Array``."""
    output_length = len(a) // 8
    if len(a) % 8 != 0:
        output_length += 1

    if a.null_count == 0 and b.null_count == 0:
        result = np.zeros(output_length, dtype=np.uint8)
        bitmap_or_unaligned(
            len(a), a.buffers()[1], a.offset, b.buffers()[1], b.offset, result
        )
        return pa.Array.from_buffers(
            pa.bool_(), len(a), [None, pa.py_buffer(result)], 0
        )
    elif a.null_count == 0:
        result = np.zeros(output_length, dtype=np.uint8)
        bitmap_or_unaligned(
            len(a), a.buffers()[1], a.offset, b.buffers()[1], b.offset, result
        )
        # b has nulls, mark all occasions of b(None) & a(True) as True -> valid_bits = a.data or b.valid_bits
        valid_bits = np.zeros(output_length, dtype=np.uint8)
        bitmap_or_unaligned(
            len(a), a.buffers()[1], a.offset, b.buffers()[0], b.offset, valid_bits
        )
        return pa.Array.from_buffers(
            pa.bool_(), len(a), [pa.py_buffer(valid_bits), pa.py_buffer(result)]
        )
        pass
    elif b.null_count == 0:
        return or_array_array(b, a)
    else:
        result = np.zeros(output_length, dtype=np.uint8)
        valid_bits = np.zeros(output_length, dtype=np.uint8)
        null_count = masked_bitmap_or_unaligned(
            len(a),
            a.buffers()[0],
            a.buffers()[1],
            a.offset,
            b.buffers()[0],
            b.buffers()[1],
            b.offset,
            result,
            valid_bits,
        )
        return pa.Array.from_buffers(
            pa.bool_(),
            len(a),
            [pa.py_buffer(valid_bits), pa.py_buffer(result)],
            null_count,
        )


@njit(locals={"value_a": numba.bool_})
def bitmap_or_unaligned_with_numpy(
    length: int,
    valid_bits_a: bytes,
    a: bytes,
    offset_a: int,
    b: np.ndarray,
    result: np.ndarray,
    valid_bits: np.ndarray,
) -> int:
    """Perform OR on a bitmap with valid bitmask and a numpy array with truthy rows."""
    null_count = 0
    for i in range(length):
        a_pos = offset_a + i
        byte_offset_a = a_pos // 8
        bit_offset_a = a_pos % 8
        mask_a = np.uint8(1 << bit_offset_a)
        valid_a = valid_bits_a[byte_offset_a] & mask_a
        value_a = a[byte_offset_a] & mask_a

        byte_offset_result = i // 8
        bit_offset_result = i % 8
        mask_result = np.uint8(1 << bit_offset_result)

        current = result[byte_offset_result]
        current_valid = valid_bits[byte_offset_result]
        if b[i] or (valid_a and value_a):
            result[byte_offset_result] = current | mask_result
            valid_bits[byte_offset_result] = current_valid | mask_result
        elif valid_a:
            result[byte_offset_result] = current & ~mask_result
            valid_bits[byte_offset_result] = current_valid | mask_result
        else:
            valid_bits[byte_offset_result] = current_valid & ~mask_result
            null_count += 1

    return null_count


@njit(locals={"value_a": numba.bool_})
def bitmap_or_unaligned_with_numpy_nonnull(
    length: int, a: bytes, offset_a: int, b: np.ndarray, result: np.ndarray
) -> None:
    """Perform OR on a bitmap and a numpy array with truthy rows."""
    for i in range(length):
        a_pos = offset_a + i
        byte_offset_a = a_pos // 8
        bit_offset_a = a_pos % 8
        mask_a = np.uint8(1 << bit_offset_a)
        value_a = a[byte_offset_a] & mask_a

        byte_offset_result = i // 8
        bit_offset_result = i % 8
        mask_result = np.uint8(1 << bit_offset_result)

        current = result[byte_offset_result]
        if b[i] or value_a:
            result[byte_offset_result] = current | mask_result
        else:
            result[byte_offset_result] = current & ~mask_result


def or_array_nparray(a: pa.Array, b: np.ndarray) -> pa.Array:
    """Perform ``pa.Array | np.ndarray``."""
    output_length = len(a) // 8
    if len(a) % 8 != 0:
        output_length += 1

    if a.null_count == 0:
        result = np.zeros(output_length, dtype=np.uint8)
        bitmap_or_unaligned_with_numpy_nonnull(
            len(a), a.buffers()[1], a.offset, b, result
        )
        return pa.Array.from_buffers(
            pa.bool_(), len(a), [None, pa.py_buffer(result)], 0
        )
    else:
        result = np.zeros(output_length, dtype=np.uint8)
        valid_bits = np.zeros(output_length, dtype=np.uint8)
        null_count = bitmap_or_unaligned_with_numpy(
            len(a), a.buffers()[0], a.buffers()[1], a.offset, b, result, valid_bits
        )
        return pa.Array.from_buffers(
            pa.bool_(),
            len(a),
            [pa.py_buffer(valid_bits), pa.py_buffer(result)],
            null_count,
        )


def or_vectorised(a: Union[pa.Array, pa.ChunkedArray], b: Any):
    """Perform OR on a boolean Arrow structure and a second operator."""
    # Scalar should be handled by or_na or all_true
    ops = {"array_array": or_array_array, "array_nparray": or_array_nparray}
    return dispatch_chunked_binary_map(a, b, ops)
