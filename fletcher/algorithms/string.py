from functools import singledispatch
from typing import Any, List, Tuple

import numpy as np
import pyarrow as pa
from numba import prange

from fletcher._algorithms import _buffer_to_view, _merge_valid_bitmaps
from fletcher._compat import njit
from fletcher.algorithms.string_builder import StringArrayBuilder, finalize_string_array
from fletcher.algorithms.utils.chunking import (
    _calculate_chunk_offsets,
    _combined_in_chunk_offsets,
    apply_per_chunk,
)


def _extract_string_buffers(arr: pa.Array) -> Tuple[np.ndarray, np.ndarray]:
    start = arr.offset
    end = arr.offset + len(arr)

    offsets = np.asanyarray(arr.buffers()[1]).view(np.int32)[start : end + 1]
    data = np.asanyarray(arr.buffers()[2]).view(np.uint8)

    return offsets, data


@njit
def _merge_string_data(
    length: int,
    valid_bits: np.ndarray,
    offsets_a: np.ndarray,
    data_a: np.ndarray,
    offsets_b: np.ndarray,
    data_b: np.ndarray,
    result_offsets: np.ndarray,
    result_data: np.ndarray,
) -> None:
    for i in range(length):
        byte_offset = i // 8
        bit_offset = i % 8
        mask = np.uint8(1 << bit_offset)
        valid = valid_bits[byte_offset] & mask

        if not valid:
            result_offsets[i + 1] = result_offsets[i]
        else:
            len_a = offsets_a[i + 1] - offsets_a[i]
            len_b = offsets_b[i + 1] - offsets_b[i]
            result_offsets[i + 1] = result_offsets[i] + len_a + len_b
            for j in range(len_a):
                result_data[result_offsets[i] + j] = data_a[offsets_a[i] + j]
            for j in range(len_b):
                result_data[result_offsets[i] + len_a + j] = data_b[offsets_b[i] + j]


@singledispatch
def _text_cat_chunked(a: Any, b: pa.ChunkedArray) -> pa.ChunkedArray:
    raise NotImplementedError(
        "_text_cat_chunked is only implemented for pa.Array and pa.ChunkedArray"
    )


@_text_cat_chunked.register(pa.ChunkedArray)
def _text_cat_chunked_1(a: pa.ChunkedArray, b: pa.ChunkedArray) -> pa.ChunkedArray:
    in_a_offsets, in_b_offsets = _combined_in_chunk_offsets(a, b)

    new_chunks: List[pa.Array] = []
    for a_offset, b_offset in zip(in_a_offsets, in_b_offsets):
        a_slice = a.chunk(a_offset[0])[a_offset[1] : a_offset[1] + a_offset[2]]
        b_slice = b.chunk(b_offset[0])[b_offset[1] : b_offset[1] + b_offset[2]]
        new_chunks.append(_text_cat(a_slice, b_slice))
    return pa.chunked_array(new_chunks)


@_text_cat_chunked.register(pa.Array)
def _text_cat_chunked_2(a: pa.Array, b: pa.ChunkedArray) -> pa.ChunkedArray:
    new_chunks = []
    offsets = _calculate_chunk_offsets(b)
    for chunk, offset in zip(b.iterchunks(), offsets):
        new_chunks.append(_text_cat(a[offset : offset + len(chunk)], chunk))
    return pa.chunked_array(new_chunks)


def _text_cat_chunked_mixed(a: pa.ChunkedArray, b: pa.Array) -> pa.ChunkedArray:
    new_chunks = []
    offsets = _calculate_chunk_offsets(a)
    for chunk, offset in zip(a.iterchunks(), offsets):
        new_chunks.append(_text_cat(chunk, b[offset : offset + len(chunk)]))
    return pa.chunked_array(new_chunks)


def _text_cat(a: pa.Array, b: pa.Array) -> pa.Array:
    if len(a) != len(b):
        raise ValueError("Lengths of arrays don't match")

    offsets_a, data_a = _extract_string_buffers(a)
    offsets_b, data_b = _extract_string_buffers(b)
    if len(a) > 0:
        valid = _merge_valid_bitmaps(a, b)
        result_offsets = np.empty(len(a) + 1, dtype=np.int32)
        result_offsets[0] = 0
        total_size = (offsets_a[-1] - offsets_a[0]) + (offsets_b[-1] - offsets_b[0])
        result_data = np.empty(total_size, dtype=np.uint8)
        _merge_string_data(
            len(a),
            valid,
            offsets_a,
            data_a,
            offsets_b,
            data_b,
            result_offsets,
            result_data,
        )
        buffers = [pa.py_buffer(x) for x in [valid, result_offsets, result_data]]
        return pa.Array.from_buffers(pa.string(), len(a), buffers)
    return a


@njit
def _text_contains_case_sensitive_nonnull(
    length: int, offsets: np.ndarray, data: np.ndarray, pat, output: np.ndarray
) -> None:
    for row_idx in range(length):
        str_len = offsets[row_idx + 1] - offsets[row_idx]

        contains = False
        for str_idx in range(max(0, str_len - len(pat) + 1)):
            pat_found = True
            for pat_idx in range(len(pat)):
                if data[offsets[row_idx] + str_idx + pat_idx] != pat[pat_idx]:
                    pat_found = False
                    break
            if pat_found:
                contains = True
                break

        # TODO: Set word-wise for better performance
        byte_offset_result = row_idx // 8
        bit_offset_result = row_idx % 8
        mask_result = np.uint8(1 << bit_offset_result)
        current = output[byte_offset_result]
        if contains:  # must be logical, not bit-wise as different bits may be flagged
            output[byte_offset_result] = current | mask_result
        else:
            output[byte_offset_result] = current & ~mask_result


@njit
def _text_contains_case_sensitive_nulls(
    length: int,
    valid_bits: np.ndarray,
    valid_offset: int,
    offsets: np.ndarray,
    data: np.ndarray,
    pat: bytes,
    output: np.ndarray,
) -> None:
    for row_idx in range(length):
        # Check whether the current entry is null.
        byte_offset = (row_idx + valid_offset) // 8
        bit_offset = (row_idx + valid_offset) % 8
        mask = np.uint8(1 << bit_offset)
        valid = valid_bits[byte_offset] & mask

        # We don't need to set the result for nulls, the calling code is
        # already dealing with them by zero'ing the output.
        if not valid:
            continue

        str_len = offsets[row_idx + 1] - offsets[row_idx]

        contains = False
        # Try to find the pattern at each starting position
        for str_idx in range(max(0, str_len - len(pat) + 1)):
            pat_found = True
            # Compare at the current position byte-by-byte
            for pat_idx in range(len(pat)):
                if data[offsets[row_idx] + str_idx + pat_idx] != pat[pat_idx]:
                    pat_found = False
                    break
            if pat_found:
                contains = True
                break

        # Write out the result into the bit-mask
        byte_offset_result = row_idx // 8
        bit_offset_result = row_idx % 8
        mask_result = np.uint8(1 << bit_offset_result)
        current = output[byte_offset_result]
        if contains:  # must be logical, not bit-wise as different bits may be flagged
            output[byte_offset_result] = current | mask_result
        else:
            output[byte_offset_result] = current & ~mask_result


@njit
def _shift_unaligned_bitmap(
    valid_bits: np.ndarray, valid_offset: int, length: int, output: np.ndarray
) -> None:
    for i in range(length):
        byte_offset = (i + valid_offset) // 8
        bit_offset = (i + valid_offset) % 8
        mask = np.uint8(1 << bit_offset)
        valid = valid_bits[byte_offset] & mask

        byte_offset_result = i // 8
        bit_offset_result = i % 8
        mask_result = np.uint8(1 << bit_offset_result)
        current = output[byte_offset_result]
        if valid:
            output[byte_offset_result] = current | mask_result


def shift_unaligned_bitmap(
    valid_buffer: pa.Buffer, offset: int, length: int
) -> pa.Buffer:
    """Shift an unaligned bitmap to be offsetted at 0."""
    output_size = length // 8
    if length % 8 > 0:
        output_size += 1
    output = np.zeros(output_size, dtype=np.uint8)

    _shift_unaligned_bitmap(valid_buffer, offset, length, output)

    return pa.py_buffer(output)


@apply_per_chunk
def _text_contains_case_sensitive(data: pa.Array, pat: str) -> pa.Array:
    """
    Check for each element in the data whether it contains the pattern ``pat``.

    This implementation does basic byte-by-byte comparison and is independent
    of any locales or encodings.
    """
    # Convert to UTF-8 bytes
    pat_bytes: bytes = pat.encode()

    # Initialise boolean (bit-packaed) output array.
    output_size = len(data) // 8
    if len(data) % 8 > 0:
        output_size += 1
    output = np.empty(output_size, dtype=np.uint8)
    if len(data) % 8 > 0:
        # Zero trailing bits
        output[-1] = 0

    offsets, data_buffer = _extract_string_buffers(data)

    if data.null_count == 0:
        valid_buffer = None
        _text_contains_case_sensitive_nonnull(
            len(data), offsets, data_buffer, pat_bytes, output
        )
    else:
        valid = _buffer_to_view(data.buffers()[0])
        _text_contains_case_sensitive_nulls(
            len(data), valid, data.offset, offsets, data_buffer, pat_bytes, output
        )
        valid_buffer = data.buffers()[0].slice(data.offset // 8)
        if data.offset % 8 != 0:
            valid_buffer = shift_unaligned_bitmap(
                valid_buffer, data.offset % 8, len(data)
            )

    return pa.Array.from_buffers(
        pa.bool_(), len(data), [valid_buffer, pa.py_buffer(output)], data.null_count
    )


@njit
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


@njit
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

        out[offset + i] = 1
        for j in range(needle_length):
            if sa.get_byte(i, string_length - needle_length + j) != needle.get_byte(j):
                out[offset + i] = 0
                break


@apply_per_chunk
def _slice_handle_chunk(pa_arr, start, end, step):
    """Slice each string according to the (start, end, step) inputs."""
    offsets, data = _extract_string_buffers(pa_arr)
    valid = _buffer_to_view(pa_arr.buffers()[0])
    if step == 0:
        raise ValueError("step cannot be zero.")

    if start >= 0 and (end is None or end >= 0) and step >= 1:
        if step == 1:
            res = _slice_pos_inputs_nostep(
                offsets, data, valid, pa_arr.offset, start, end
            )
        else:
            res = _slice_pos_inputs_step(
                offsets, data, valid, pa_arr.offset, start, end, step
            )
    else:
        res = _slice_generic(offsets, data, valid, pa_arr.offset, start, end, step)

    return finalize_string_array(res, pa.string())


@njit
def get_utf8_size(first_byte: int):
    if first_byte < 0b10000000:
        return 1
    elif first_byte < 0b11100000:
        return 2
    elif first_byte < 0b11110000:
        return 3
    else:
        return 4


@njit
def _check_valid(valid_bits, i, valid_offset) -> bool:
    byte_offset = (i + valid_offset) // 8
    bit_offset = (i + valid_offset) % 8
    mask = np.uint8(1 << bit_offset)
    return valid_bits[byte_offset] & mask


@njit
def _slice_pos_inputs_nostep(
    offsets, data, valid_bits, valid_offset, start: int, end: int
) -> StringArrayBuilder:
    """
    start, end >= 0
    step == 1
    """
    builder = StringArrayBuilder(len(offsets) - 1)

    for i in prange(len(offsets) - 1):

        if len(valid_bits) > 0:
            valid = _check_valid(valid_bits, i, valid_offset)
            if not valid:
                builder.append_null()
                continue

        str_len_bytes = offsets[i + 1] - offsets[i]

        char_idx = 0
        byte_idx = 0

        while char_idx < start and byte_idx < str_len_bytes:
            char_idx += 1
            byte_idx += get_utf8_size(data[offsets[i] + byte_idx])

        start_byte = offsets[i] + byte_idx

        while (end is None or char_idx < end) and byte_idx < str_len_bytes:
            char_idx += 1
            byte_idx += get_utf8_size(data[offsets[i] + byte_idx])

        end_byte = offsets[i] + byte_idx
        builder.append_value(data[start_byte:end_byte], end_byte - start_byte)
    return builder


@njit
def _slice_pos_inputs_step(
    offsets, data, valid_bits, valid_offset, start: int, end: int, step: int
) -> StringArrayBuilder:
    """
    start, end >= 0
    step > 1
    """
    builder = StringArrayBuilder(len(offsets) - 1)

    for i in prange(len(offsets) - 1):
        if len(valid_bits) > 0:
            valid = _check_valid(valid_bits, i, valid_offset)
            if not valid:
                builder.append_null()
                continue

        str_len_bytes = offsets[i + 1] - offsets[i]

        char_idx = 0
        byte_idx = 0

        while char_idx < start and byte_idx < str_len_bytes:
            char_idx += 1
            byte_idx += get_utf8_size(data[offsets[i] + byte_idx])

        to_skip = 0
        include_bytes: List[bytes] = []

        while (end is None or char_idx < end) and byte_idx < str_len_bytes:
            char_size = get_utf8_size(data[offsets[i] + byte_idx])

            if not to_skip:
                include_bytes.extend(
                    data[offsets[i] + byte_idx : offsets[i] + byte_idx + char_size]
                )
                to_skip = step

            char_idx += 1
            byte_idx += char_size
            to_skip -= 1

        builder.append_value(include_bytes, len(include_bytes))
        # print(include_bytes)
        # print(len(include_bytes))
    return builder


@njit
def _slice_generic(
    offsets, data, valid_bits, valid_offset, start: int, end: int, step: int
) -> StringArrayBuilder:
    builder = StringArrayBuilder(len(offsets) - 1)

    for i in prange(len(offsets) - 1):
        if len(valid_bits) > 0:
            valid = _check_valid(valid_bits, i, valid_offset)
            if not valid:
                builder.append_null()
                continue

        str_len_bytes = offsets[i + 1] - offsets[i]
        char_bytes: List[bytes] = []
        byte_idx = 0

        while byte_idx < str_len_bytes:
            char_size = get_utf8_size(data[offsets[i] + byte_idx])
            char_bytes.append(
                data[offsets[i] + byte_idx : offsets[i] + byte_idx + char_size]
            )
            byte_idx += char_size

        include_bytes: List[bytes] = []  # type: ignore

        char_idx = start
        if start >= -len(char_bytes) and start < 0:
            char_idx += len(char_bytes)

        true_end = end
        if end >= -len(char_bytes) and end < 0:
            true_end += len(char_bytes)

        # Positive step
        if step > 0:
            if char_idx < 0:
                char_idx = 0
            while (end is None or char_idx < true_end) and char_idx < len(char_bytes):
                include_bytes.extend(char_bytes[char_idx])  # type: ignore
                char_idx += step

        # Negative step
        else:
            if char_idx >= len(char_bytes):
                char_idx = len(char_bytes) - 1
            while (end is None or char_idx > true_end) and char_idx >= 0:
                if char_idx < len(char_bytes):
                    include_bytes.extend(char_bytes[char_idx])  # type: ignore
                char_idx += step

        builder.append_value(include_bytes, len(include_bytes))

    return builder
