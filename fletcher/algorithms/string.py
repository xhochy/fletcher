from functools import singledispatch
from typing import Any, List, Tuple

import numpy as np
import pyarrow as pa

from fletcher._algorithms import _buffer_to_view, _merge_valid_bitmaps
from fletcher._compat import njit
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
    length: int, offsets: np.ndarray, data: np.ndarray, pat: bytes, output: np.ndarray
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
def _check_valid_row(
    row_idx: int, valid_bits: np.ndarray, valid_offset: int
) -> bool:
    """ Check whether the current entry is null. """
    byte_offset = (row_idx + valid_offset) // 8
    bit_offset = (row_idx + valid_offset) % 8
    mask = np.uint8(1 << bit_offset)
    valid = valid_bits[byte_offset] & mask
    return valid


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
        # We don't need to set the result for nulls, the calling code is
        # already dealing with them by zero'ing the output.
        if not _check_valid_row(row_idx, valid_bits, valid_offset):
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
def _compute_kmp_failure_function(
    pat: bytes
) -> np.ndarray:
    """
    \texttt{f[i]} is length of the longest proper suffix
    of the $i$-th prefix of $pat$ that is a prefix of $pat$
    """

    length = len(pat)
    f = np.empty(length + 1, dtype=np.int32)

    f[0] = -1
    for i in range(1, length + 1):
        f[i] = f[i - 1]
        while f[i] != -1 and pat[f[i]] != pat[i - 1]:
            f[i] = f[f[i]]
        f[i] += 1

    return f


@njit
def _text_replace_case_sensitive_nonnull(
    length: int,
    offsets: np.ndarray,
    data: np.ndarray,
    pat: bytes,
    repl: bytes,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Instead of using a StringBuilder, we make two passes:
     The first one computes the offsets for the output buffer.
     The second one actually does the replace in the buffer.
    """

    failure_function = _compute_kmp_failure_function(pat)

    # Computes output buffer offsets
    output_offsets = np.empty(length + 1, dtype=np.int32)
    cumulative_offset = 0
    for row_idx in range(length):
        output_offsets[row_idx] = cumulative_offset
        matched_until = 0
        matches_done = 0
        for str_idx in range(offsets[row_idx], offsets[row_idx + 1]):
            while matched_until != -1 and pat[matched_until] != data[str_idx]:
                matched_until = failure_function[matched_until]

            cumulative_offset += 1
            matched_until += 1
            if matched_until == len(pat) and matches_done < n:
                cumulative_offset += len(repl) - len(pat)
                matches_done += 1
                matched_until = 0

    output_offsets[length] = cumulative_offset

    # Replace in the output_buffer
    output_buffer = np.empty(cumulative_offset, dtype=np.uint8)
    for row_idx in range(length):
        matched_until = 0
        matches_done = 0
        pos_output = output_offsets[row_idx]
        for str_idx in range(offsets[row_idx], offsets[row_idx + 1]):
            while matched_until != -1 and pat[matched_until] != data[str_idx]:
                matched_until = failure_function[matched_until]

            output_buffer[pos_output] = data[str_idx]
            pos_output += 1
            matched_until += 1

            if matched_until == len(pat) and matches_done < n:
                pos_output -= len(pat)
                for i in range(len(repl)):
                    output_buffer[pos_output] = repl[i]
                    pos_output += 1
                matches_done += 1
                matched_until = 0

    return output_buffer, output_offsets


@njit
def _text_replace_case_sensitive_nulls(
    length: int,
    valid_bits: np.ndarray,
    valid_offset: int,
    offsets: np.ndarray,
    data: np.ndarray,
    pat: bytes,
    repl: bytes,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TODO:
    """

    failure_function = _compute_kmp_failure_function(pat)

    # Computes output buffer offsets
    output_offsets = np.empty(length + 1, dtype=np.int32)
    cumulative_offset = 0
    for row_idx in range(length):
        output_offsets[row_idx] = cumulative_offset
        if not _check_valid_row(row_idx, valid_bits, valid_offset):
            continue

        matched_until = 0
        matches_done = 0
        for str_idx in range(offsets[row_idx], offsets[row_idx + 1]):
            while matched_until != -1 and pat[matched_until] != data[str_idx]:
                matched_until = failure_function[matched_until]

            cumulative_offset += 1
            matched_until += 1
            if matched_until == len(pat) and matches_done < n:
                cumulative_offset += len(repl) - len(pat)
                matches_done += 1
                matched_until = 0

    output_offsets[length] = cumulative_offset

    # Replace in the output_buffer
    output_buffer = np.empty(cumulative_offset, dtype=np.uint8)
    for row_idx in range(length):
        if not _check_valid_row(row_idx, valid_bits, valid_offset):
            continue

        matched_until = 0
        matches_done = 0
        pos_output = output_offsets[row_idx]
        for str_idx in range(offsets[row_idx], offsets[row_idx + 1]):
            while matched_until != -1 and pat[matched_until] != data[str_idx]:
                matched_until = failure_function[matched_until]

            output_buffer[pos_output] = data[str_idx]
            pos_output += 1
            matched_until += 1

            if matched_until == len(pat) and matches_done < n:
                pos_output -= len(pat)
                for i in range(len(repl)):
                    output_buffer[pos_output] = repl[i]
                    pos_output += 1
                matches_done += 1
                matched_until = 0

    return output_buffer, output_offsets


@apply_per_chunk
def _text_replace_case_sensitive(data: pa.Array, pat: str, repl: str, n: int) -> pa.Array:
    """
    TODO:
    """

    # Convert to UTF-8 bytes
    pat_bytes: bytes = pat.encode()
    repl_bytes: bytes = repl.encode()

    offsets, data_buffer = _extract_string_buffers(data)

    if data.null_count == 0:
        valid_buffer = None
        output_buffer, output_offsets = _text_replace_case_sensitive_nonnull(
            len(data), offsets, data_buffer, pat_bytes, repl_bytes, n
        )
    else:
        valid_buffer = data.buffers()[0].slice(data.offset // 8)
        if data.offset % 8 != 0:
            valid_buffer = shift_unaligned_bitmap(
                valid_buffer, data.offset % 8, len(data)
            )
        valid = _buffer_to_view(data.buffers()[0])
        output_buffer, output_offsets = _text_replace_case_sensitive_nulls(
            len(data), valid, data.offset, pat_bytes, n
        )

    return pa.Array.from_buffers(
        pa.string(), len(data), [valid_buffer, ], data.null_count
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
