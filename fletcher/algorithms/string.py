from functools import singledispatch
from typing import Any, Callable, List, Tuple, Union

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
from fletcher.algorithms.utils.kmp import compute_kmp_failure_function


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
def _check_valid_row(row_idx: int, valid_bits: np.ndarray, valid_offset: int) -> bool:
    """ Check whether the current entry is null. """
    byte_offset = (row_idx + valid_offset) // 8
    bit_offset = (row_idx + valid_offset) % 8
    mask = np.uint8(1 << bit_offset)
    valid = valid_bits[byte_offset] & mask
    return valid


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


@njit
def _text_count_case_sensitive_numba(
    length: int,
    valid_bits: np.ndarray,
    valid_offset: int,
    offsets: np.ndarray,
    data: np.ndarray,
    pat: bytes,
) -> np.ndarray:
    failure_function = compute_kmp_failure_function(pat)

    output = np.empty(length, dtype=np.int64)

    has_nulls = valid_bits.size > 0

    for row_idx in range(length):
        if has_nulls and not _check_valid_row(row_idx, valid_bits, valid_offset):
            continue

        matched_len = 0
        output[row_idx] = 0

        if len(pat) == 0:
            output[row_idx] = offsets[row_idx + 1] - offsets[row_idx] + 1
            continue

        for str_idx in range(offsets[row_idx], offsets[row_idx + 1]):
            # Manually inlined utils.kmp.append_to_kmp_matching for performance
            while matched_len > -1 and pat[matched_len] != data[str_idx]:
                matched_len = failure_function[matched_len]
            matched_len = matched_len + 1

            if matched_len == len(pat):
                output[row_idx] += 1
                # `matched_len=0` ensures overlapping matches are not counted.
                # This matches the behavior of Python's builtin `count`
                # function.
                matched_len = 0

    return output


@apply_per_chunk
def _text_count_case_sensitive(data: pa.Array, pat: str) -> pa.Array:
    """
    For each row in the data computes the number of occurrences of the pattern ``pat``.
    This implementation does basic byte-by-byte comparison and is independent
    of any locales or encodings.
    """

    # Convert to UTF-8 bytes
    pat_bytes: bytes = pat.encode()

    offsets_buffer, data_buffer = _extract_string_buffers(data)

    if data.null_count == 0:
        valid_buffer = np.empty(0, dtype=np.uint8)
    else:
        valid_buffer = _buffer_to_view(data.buffers()[0])

    output = _text_count_case_sensitive_numba(
        len(data), valid_buffer, data.offset, offsets_buffer, data_buffer, pat_bytes
    )

    if data.null_count == 0:
        output_valid = None
    else:
        output_valid = data.buffers()[0].slice(data.offset // 8)
        if data.offset % 8 != 0:
            output_valid = shift_unaligned_bitmap(
                output_valid, data.offset % 8, len(data)
            )

    buffers = [output_valid, pa.py_buffer(output)]
    return pa.Array.from_buffers(pa.int64(), len(data), buffers, data.null_count)


@njit
def _text_contains_case_sensitive_numba(
    length: int,
    valid_bits: np.ndarray,
    valid_offset: int,
    offsets: np.ndarray,
    data: np.ndarray,
    pat: bytes,
) -> np.ndarray:
    failure_function = compute_kmp_failure_function(pat)

    # Initialise boolean (bit-packaed) output array.
    output_size = length // 8
    if length % 8 > 0:
        output_size += 1
    output = np.empty(output_size, dtype=np.uint8)

    if length % 8 > 0:
        # Zero trailing bits
        output[-1] = 0

    has_nulls = valid_bits.size > 0

    for row_idx in range(length):
        if has_nulls and not _check_valid_row(row_idx, valid_bits, valid_offset):
            continue

        matched_len = 0
        contains = False
        for str_idx in range(offsets[row_idx], offsets[row_idx + 1]):
            if matched_len == len(pat):
                contains = True
                break

            # Manually inlined utils.kmp.append_to_kmp_matching for
            # performance
            while matched_len > -1 and pat[matched_len] != data[str_idx]:
                matched_len = failure_function[matched_len]
            matched_len = matched_len + 1

        if matched_len == len(pat):
            contains = True

        # Write out the result into the bit-mask
        byte_offset_result = row_idx // 8
        bit_offset_result = row_idx % 8
        mask_result = np.uint8(1 << bit_offset_result)
        current = output[byte_offset_result]
        if contains:  # must be logical, not bit-wise as different bits may be flagged
            output[byte_offset_result] = current | mask_result
        else:
            output[byte_offset_result] = current & ~mask_result

    return output


@apply_per_chunk
def _text_contains_case_sensitive(data: pa.Array, pat: str) -> pa.Array:
    """
    Check for each element in the data whether it contains the pattern ``pat``.

    This implementation does basic byte-by-byte comparison and is independent
    of any locales or encodings.
    """
    # Convert to UTF-8 bytes
    pat_bytes: bytes = pat.encode()

    offsets_buffer, data_buffer = _extract_string_buffers(data)

    if data.null_count == 0:
        valid_buffer = np.empty(0, dtype=np.uint8)
    else:
        valid_buffer = _buffer_to_view(data.buffers()[0])

    output = _text_contains_case_sensitive_numba(
        len(data), valid_buffer, data.offset, offsets_buffer, data_buffer, pat_bytes
    )

    if data.null_count == 0:
        output_valid = None
    else:
        output_valid = data.buffers()[0].slice(data.offset // 8)
        if data.offset % 8 != 0:
            output_valid = shift_unaligned_bitmap(
                output_valid, data.offset % 8, len(data)
            )

    buffers = [output_valid, pa.py_buffer(output)]
    return pa.Array.from_buffers(pa.bool_(), len(data), buffers, data.null_count)


@njit
def _text_replace_case_sensitive_empty_pattern(
    length: int,
    valid_bits: np.ndarray,
    valid_offset: int,
    offsets: np.ndarray,
    data: np.ndarray,
    repl: bytes,
    max_repl: int,
):
    """
    A special case for replace when pat=''.

    Assumes max_repl != 0.
    """
    output_offsets = np.empty(length + 1, dtype=np.int32)
    cumulative_offset = 0

    has_nulls = valid_bits.size > 0

    for row_idx in range(length):
        output_offsets[row_idx] = cumulative_offset

        if has_nulls and not _check_valid_row(row_idx, valid_bits, valid_offset):
            continue

        row_len = offsets[row_idx + 1] - offsets[row_idx]

        if max_repl < 0:
            matches_done = row_len + 1
        else:
            matches_done = min(max_repl, row_len + 1)

        cumulative_offset += row_len + matches_done * len(repl)

    output_offsets[length] = cumulative_offset

    # Replace in the output_buffer
    output_buffer = np.empty(cumulative_offset, dtype=np.uint8)
    output_pos = 0
    for row_idx in range(length):
        if has_nulls and not _check_valid_row(row_idx, valid_bits, valid_offset):
            continue

        matches_done = 0

        if max_repl != 0:
            matches_done += 1
            for char in repl:
                output_buffer[output_pos] = char
                output_pos += 1

        for str_idx in range(offsets[row_idx], offsets[row_idx + 1]):
            output_buffer[output_pos] = data[str_idx]
            output_pos += 1

            if matches_done != max_repl:
                matches_done += 1
                for char in repl:
                    output_buffer[output_pos] = char
                    output_pos += 1

    return output_offsets, output_buffer


@njit
def _text_replace_case_sensitive_numba(
    length: int,
    valid_bits: np.ndarray,
    valid_offset: int,
    offsets: np.ndarray,
    data: np.ndarray,
    pat: bytes,
    repl: bytes,
    max_repl: int,
) -> Tuple[np.ndarray, np.ndarray]:

    failure_function = compute_kmp_failure_function(pat)

    # Computes output buffer offsets
    output_offsets = np.empty(length + 1, dtype=np.int32)
    cumulative_offset = 0

    has_nulls = valid_bits.size > 0
    match_len_change = len(repl) - len(pat)

    for row_idx in range(length):
        output_offsets[row_idx] = cumulative_offset

        if has_nulls and not _check_valid_row(row_idx, valid_bits, valid_offset):
            continue

        row_len = offsets[row_idx + 1] - offsets[row_idx]
        cumulative_offset += row_len

        matched_len = 0
        matches_done = 0

        for str_idx in range(offsets[row_idx], offsets[row_idx + 1]):
            # Manually inlined utils.kmp.append_to_kmp_matching for performance
            while matched_len > -1 and pat[matched_len] != data[str_idx]:
                matched_len = failure_function[matched_len]
            matched_len = matched_len + 1

            if matched_len == len(pat):
                matches_done += 1
                matched_len = 0
                if matches_done == max_repl:
                    break

        cumulative_offset += match_len_change * matches_done

    output_offsets[length] = cumulative_offset

    output_buffer = np.empty(cumulative_offset, dtype=np.uint8)
    output_pos = 0
    for row_idx in range(length):
        if has_nulls and not _check_valid_row(row_idx, valid_bits, valid_offset):
            continue

        matched_len = 0
        matches_done = 0

        write_idx = offsets[row_idx]
        for read_idx in range(offsets[row_idx], offsets[row_idx + 1]):
            # A modified version of utils.kmp.append_to_kmp_matching
            while matched_len > -1 and pat[matched_len] != data[read_idx]:
                matched_len = failure_function[matched_len]
            matched_len = matched_len + 1

            if read_idx - write_idx == len(pat):
                output_buffer[output_pos] = data[write_idx]
                output_pos += 1
                write_idx += 1

            if matched_len == len(pat):
                matched_len = 0
                if matches_done != max_repl:
                    matches_done += 1
                    write_idx = read_idx + 1

                    for char in repl:
                        output_buffer[output_pos] = char
                        output_pos += 1

        while write_idx < offsets[row_idx + 1]:
            output_buffer[output_pos] = data[write_idx]
            output_pos += 1
            write_idx += 1

    return output_offsets, output_buffer


@apply_per_chunk
def _text_replace_case_sensitive(
    data: pa.Array, pat: str, repl: str, max_repl: int
) -> pa.Array:
    """
    Replace occurrences of ``pat`` with ``repl`` in the Series/Index with some other string. For every
    row, only the first ``max_repl`` replacements will be performed. If ``max_repl = -1`` we consider that
    we have no limit for the number of replacements.

    This implementation does basic byte-by-byte comparison and is independent
    of any locales or encodings.
    """

    # Convert to UTF-8 bytes
    pat_bytes: bytes = pat.encode()
    repl_bytes: bytes = repl.encode()

    offsets_buffer, data_buffer = _extract_string_buffers(data)

    if data.null_count == 0:
        valid_buffer = np.empty(0, dtype=np.uint8)
    else:
        valid_buffer = _buffer_to_view(data.buffers()[0])

    if len(pat) > 0:
        output_t = _text_replace_case_sensitive_numba(
            len(data),
            valid_buffer,
            data.offset,
            offsets_buffer,
            data_buffer,
            pat_bytes,
            repl_bytes,
            max_repl,
        )
    else:
        output_t = _text_replace_case_sensitive_empty_pattern(
            len(data),
            valid_buffer,
            data.offset,
            offsets_buffer,
            data_buffer,
            repl_bytes,
            max_repl,
        )

    output_offsets, output_buffer = output_t

    if data.null_count == 0:
        output_valid = None
    else:
        output_valid = data.buffers()[0].slice(data.offset // 8)
        if data.offset % 8 != 0:
            output_valid = shift_unaligned_bitmap(
                output_valid, data.offset % 8, len(data)
            )

    buffers = [output_valid, pa.py_buffer(output_offsets), pa.py_buffer(output_buffer)]
    return pa.Array.from_buffers(pa.string(), len(data), buffers, data.null_count)


@apply_per_chunk
def _text_strip(data: pa.Array, to_strip) -> pa.Array:
    """
    Strip the characters of ``to_strip`` from start and end of each element in the data.
    """
    if len(data) == 0:
        return data

    offsets, data_buffer = _extract_string_buffers(data)

    valid_buffer = data.buffers()[0]
    valid_offset = data.offset
    builder = StringArrayBuilder(max(len(data_buffer), len(data)))

    _do_strip(
        valid_buffer,
        valid_offset,
        offsets,
        data_buffer,
        len(data),
        to_strip,
        inout_builder=builder,
    )

    result_array = finalize_string_array(builder, pa.string())
    return result_array


@njit
def _utf8_chr4(arr):
    return chr(
        np.int32((arr[0] & 0x7) << 18)
        | np.int32((arr[1] & 0x3F) << 12)
        | np.int32((arr[2] & 0x3F) << 6)
        | np.int32((arr[3] & 0x3F))
    )


@njit
def _utf8_chr3(arr):
    return chr(
        np.int32((arr[0] & 0xF)) << 12
        | np.int32((arr[1] & 0x3F) << 6)
        | np.int32((arr[2] & 0x3F))
    )


@njit
def _utf8_chr2(arr):
    return chr(np.int32((arr[0] & 0x1F)) << 6 | np.int32((arr[1] & 0x3F)))


@njit
def _extract_striped_string(last_offset, offset, data_buffer, to_strip):
    if last_offset < offset:
        start_offset = last_offset
        while start_offset < offset:
            if (data_buffer[start_offset] & 0x80) == 0:
                if chr(data_buffer[start_offset]) in to_strip:
                    start_offset += 1
                else:
                    break
            # for utf-8 encoding, see: https://en.wikipedia.org/wiki/UTF-8
            elif (
                (data_buffer[start_offset] & 0xF8) == 0xF0
                and start_offset + 3 < offset
                and _utf8_chr4(data_buffer[start_offset : start_offset + 4]) in to_strip
            ):
                start_offset += 4
            elif (
                (data_buffer[start_offset] & 0xF0) == 0xE0
                and start_offset + 2 < offset
                and _utf8_chr3(data_buffer[start_offset : start_offset + 3]) in to_strip
            ):
                start_offset += 3
            elif (
                (data_buffer[start_offset] & 0xE0) == 0xC0
                and start_offset + 1 < offset
                and _utf8_chr2(data_buffer[start_offset : start_offset + 2]) in to_strip
            ):
                start_offset += 2
            else:
                break
        end_offset = offset
        while end_offset > start_offset:
            if (data_buffer[end_offset - 1] & 0x80) == 0:
                if chr(data_buffer[end_offset - 1]) in to_strip:
                    end_offset -= 1
                else:
                    break
            elif (
                end_offset > start_offset + 3
                and (data_buffer[end_offset - 4] & 0xF8) == 0xF0
                and _utf8_chr4(data_buffer[end_offset - 4 : end_offset]) in to_strip
            ):
                end_offset -= 4
            elif (
                end_offset > start_offset + 2
                and (data_buffer[end_offset - 3] & 0xF0) == 0xE0
                and _utf8_chr3(data_buffer[end_offset - 3 : end_offset]) in to_strip
            ):
                end_offset -= 3
            elif (
                end_offset > start_offset + 1
                and (data_buffer[end_offset - 2] & 0xE0) == 0xC0
                and _utf8_chr2(data_buffer[end_offset - 2 : end_offset]) in to_strip
            ):
                end_offset -= 2
            else:
                break

        stripped_str = data_buffer[start_offset:end_offset]
    else:
        stripped_str = data_buffer[0:0]
    return stripped_str


@njit
def _do_strip(
    valid_buffer, valid_offset, offsets, data_buffer, len_data, to_strip, inout_builder
):
    previous_offset = offsets[0]
    for idx in range(len_data):
        current_offset = offsets[1 + idx]
        valid = (
            bool(
                valid_buffer[(idx + valid_offset) // 8]
                & (1 << ((idx + valid_offset) % 8))
            )
            if valid_buffer is not None
            else True
        )
        if valid:
            current_str = _extract_striped_string(
                previous_offset, current_offset, data_buffer, to_strip
            )
            inout_builder.append_value(current_str, len(current_str))
        else:
            inout_builder.append_null()
        previous_offset = current_offset


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
            byte_offset = (i + valid_offset) // 8
            bit_offset = (i + valid_offset) % 8
            mask = np.uint8(1 << bit_offset)
            valid = valid_bits[byte_offset] & mask
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
            byte_offset = (i + valid_offset) // 8
            bit_offset = (i + valid_offset) % 8
            mask = np.uint8(1 << bit_offset)
            valid = valid_bits[byte_offset] & mask
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
    return builder


@njit
def _slice_generic(
    offsets, data, valid_bits, valid_offset, start: int, end: int, step: int
) -> StringArrayBuilder:
    builder = StringArrayBuilder(len(offsets) - 1)

    for i in prange(len(offsets) - 1):
        if len(valid_bits) > 0:
            byte_offset = (i + valid_offset) // 8
            bit_offset = (i + valid_offset) % 8
            mask = np.uint8(1 << bit_offset)
            valid = valid_bits[byte_offset] & mask
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


@njit
def _apply_no_nulls(
    func: Callable,
    length: int,
    offsets_buffer_a,
    data_buffer_a,
    offsets_buffer_b,
    data_buffer_b,
    out,
):
    for i in range(length):
        out[i] = func(
            data_buffer_a[offsets_buffer_a[i] :],
            offsets_buffer_a[i + 1] - offsets_buffer_a[i],
            data_buffer_b[offsets_buffer_b[i] :],
            offsets_buffer_b[i + 1] - offsets_buffer_b[i],
        )


@njit
def _apply_with_nulls(
    func: Callable,
    length: int,
    valid,
    offsets_buffer_a,
    data_buffer_a,
    offsets_buffer_b,
    data_buffer_b,
    out,
):
    for i in range(length):
        # Check if one of the entries is null
        byte_offset = i // 8
        bit_offset = i % 8
        mask = np.uint8(1 << bit_offset)
        is_valid = valid[byte_offset] & mask

        if is_valid:
            out[i] = func(
                data_buffer_a[offsets_buffer_a[i] :],
                offsets_buffer_a[i + 1] - offsets_buffer_a[i],
                data_buffer_b[offsets_buffer_b[i] :],
                offsets_buffer_b[i + 1] - offsets_buffer_b[i],
            )


@njit(parallel=True)
def _apply_no_nulls_parallel(
    func: Callable,
    length: int,
    offsets_buffer_a,
    data_buffer_a,
    offsets_buffer_b,
    data_buffer_b,
    out,
):
    for i in prange(length):
        out[i] = func(
            data_buffer_a[offsets_buffer_a[i] :],
            offsets_buffer_a[i + 1] - offsets_buffer_a[i],
            data_buffer_b[offsets_buffer_b[i] :],
            offsets_buffer_b[i + 1] - offsets_buffer_b[i],
        )


@njit(parallel=True)
def _apply_with_nulls_parallel(
    func: Callable,
    length: int,
    valid,
    offsets_buffer_a,
    data_buffer_a,
    offsets_buffer_b,
    data_buffer_b,
    out,
):
    for i in prange(length):
        # Check if one of the entries is null
        byte_offset = i // 8
        bit_offset = i % 8
        mask = np.uint8(1 << bit_offset)
        is_valid = valid[byte_offset] & mask

        if is_valid:
            out[i] = func(
                data_buffer_a[offsets_buffer_a[i] :],
                offsets_buffer_a[i + 1] - offsets_buffer_a[i],
                data_buffer_b[offsets_buffer_b[i] :],
                offsets_buffer_b[i + 1] - offsets_buffer_b[i],
            )


def _apply_binary_str_array(
    a: pa.Array, b: pa.Array, *, func: Callable, output_dtype, parallel: bool = False
):
    out = np.empty(len(a), dtype=output_dtype)

    offsets_buffer_a, data_buffer_a = _extract_string_buffers(a)
    offsets_buffer_b, data_buffer_b = _extract_string_buffers(b)

    if a.null_count == 0 and b.null_count == 0:
        if parallel:
            call = _apply_no_nulls_parallel
        else:
            call = _apply_no_nulls
        call(
            func,
            len(a),
            offsets_buffer_a,
            data_buffer_a,
            offsets_buffer_b,
            data_buffer_b,
            out,
        )
        return pa.array(out)
    else:
        valid = _merge_valid_bitmaps(a, b)
        if parallel:
            call = _apply_with_nulls_parallel
        else:
            call = _apply_with_nulls
        call(
            func,
            len(a),
            valid,
            offsets_buffer_a,
            data_buffer_a,
            offsets_buffer_b,
            data_buffer_b,
            out,
        )
        buffers = [pa.py_buffer(x) for x in [valid, out]]
        return pa.Array.from_buffers(pa.int64(), len(out), buffers)


def apply_binary_str(
    a: Union[pa.Array, pa.ChunkedArray],
    b: Union[pa.Array, pa.ChunkedArray],
    *,
    func: Callable,
    output_dtype,
    parallel: bool = False,
):
    """
    Apply an element-wise numba-jitted function on two Arrow columns.

    The supplied function must return a numpy-compatible scalar.
    Handling of missing data and chunking of the inputs is done automatically.
    """
    if len(a) != len(b):
        raise ValueError("Inputs don't have the same length.")

    if isinstance(a, pa.ChunkedArray):
        if isinstance(b, pa.ChunkedArray):
            in_a_offsets, in_b_offsets = _combined_in_chunk_offsets(a, b)

            new_chunks: List[pa.Array] = []
            for a_offset, b_offset in zip(in_a_offsets, in_b_offsets):
                a_slice = a.chunk(a_offset[0])[a_offset[1] : a_offset[1] + a_offset[2]]
                b_slice = b.chunk(b_offset[0])[b_offset[1] : b_offset[1] + b_offset[2]]
                new_chunks.append(
                    _apply_binary_str_array(
                        a_slice,
                        b_slice,
                        func=func,
                        output_dtype=output_dtype,
                        parallel=parallel,
                    )
                )
            return pa.chunked_array(new_chunks)
        elif isinstance(b, pa.Array):
            new_chunks = []
            offsets = _calculate_chunk_offsets(a)
            for chunk, offset in zip(a.iterchunks(), offsets):
                new_chunks.append(
                    _apply_binary_str_array(
                        chunk,
                        b[offset : offset + len(chunk)],
                        func=func,
                        output_dtype=output_dtype,
                        parallel=parallel,
                    )
                )
            return pa.chunked_array(new_chunks)
        else:
            raise ValueError(f"left operand has unsupported type {type(b)}")
    elif isinstance(a, pa.Array):
        if isinstance(b, pa.ChunkedArray):
            new_chunks = []
            offsets = _calculate_chunk_offsets(b)
            for chunk, offset in zip(b.iterchunks(), offsets):
                new_chunks.append(
                    _apply_binary_str_array(
                        a[offset : offset + len(chunk)],
                        chunk,
                        func=func,
                        output_dtype=output_dtype,
                        parallel=parallel,
                    )
                )
            return pa.chunked_array(new_chunks)
        elif isinstance(b, pa.Array):
            return _apply_binary_str_array(
                a, b, func=func, output_dtype=output_dtype, parallel=parallel
            )
        else:
            raise ValueError(f"left operand has unsupported type {type(b)}")
    else:
        raise ValueError(f"left operand has unsupported type {type(a)}")
