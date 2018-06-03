# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numba
import numpy as np

from ._numba_compat import NumbaStringArray


@numba.jit(nogil=True, nopython=True)
def _extract_isnull_bytemap(bitmap, bitmap_length, bitmap_offset, dst_offset, dst):
    """
    (internal) write the values of a valid bitmap as bytes to a pre-allocatored
    isnull bytemap.

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


def extract_isnull_bytemap(chunked_array):
    """
    Extract the valid bitmaps of a chunked array into numpy isnull bytemaps.

    Parameters
    ----------
    chunked_array: pyarrow.ChunkedArray

    Returns
    -------
    valid_bytemap: numpy.array
    """
    # TODO: Can we use np.empty here to improve performance?
    result = np.zeros(len(chunked_array), dtype=bool)

    offset = 0
    for chunk in chunked_array.chunks:
        valid_bitmap = chunk.buffers()[0]
        if valid_bitmap:
            # TODO(ARROW-2664): We only need to following line to support
            #   executing the code in disabled-JIT mode.
            buf = memoryview(valid_bitmap)
            _extract_isnull_bytemap(buf, len(chunk), chunk.offset, offset, result)
        else:
            raise NotImplementedError()
            # _fill_bytemap(
        offset += len(chunk)

    return result


@numba.jit(nogil=True, nopython=True)
def _argquicksort_string_array(indices, sa, lo, hi):
    """
    Textbook implementation of quicksort.

    This is modeled after https://en.wikipedia.org/wiki/Quicksort#Lomuto_partition_scheme,
    thus this is probably not the most performant variant. If you're into
    algorithms and want to get some performance, this might be a nice task.
    """
    if lo < hi:
        p = _argquickpartition_string_array(indices, sa, lo, hi)
        _argquicksort_string_array(indices, sa, lo, p - 1)
        _argquicksort_string_array(indices, sa, p + 1, hi)


@numba.jit(nogil=True, nopython=True)
def _argquickpartition_string_array(indices, sa, lo, hi):
    """
    Partition part of quicksort.

    "Randomly" chose a pivot element p; ensure that all elements smaller than
    p are in a lower index than p and all greater or equal (in the case that
    there is another element equal to p) in higher indices.
    """
    pivot = indices[hi]
    i = lo - 1
    for j in range(lo, hi):
        if sa.elements_lt(indices[j], pivot):
            i += 1
            tmp = indices[i]
            indices[i] = indices[j]
            indices[j] = tmp
    tmp = indices[i + 1]
    indices[i + 1] = indices[hi]
    indices[hi] = tmp
    return i + 1


@numba.jit(nogil=True, nopython=True)
def _argsort_string_array_pandas(sa, indices, non_na_indices, result):
    """
    Run pandas-like argsort on a pyarrow.StringArray.

    In contrast to NumPy, na-values will simply be skipped instead of being
    included in the sorting process.

    Parameters
    ----------
    sa: fletcher._numba_compat.NumbaStringArray
    indices: numpy.ndarray[int64]
    non_na_indices: numpy.ndarray[int64]
        Temporay array that holds the indices as they will be returned for
        Pandas, i.e. the index that would be retrieved by skipping the NA
        entries.
    """
    # Step 1: Intially fill indices and non_na_indices
    #   * indices will contain all indices for entries that are non-null, i.e.
    #     it has as many entries as the array has valid values.
    #   * non_na_indices will be will filled with the indices as if the null
    #     entries were omitted in the array. Null entries are represented as -1
    #     in this array.
    non_na_index = 0
    for i in range(sa.size):
        if not sa.isnull(i):
            non_na_indices[i] = -1
        else:
            indices[non_na_index] = i
            non_na_indices[i] = non_na_index
            non_na_index += 1

    # Step 2: Recursively run quicksort, thereby sort indices but use
    #   the array for comparison.
    _argquicksort_string_array(indices, sa, 0, non_na_index - 1)

    # Step 3: The indices are now sorted but we need to map them to their
    #   non_na_indices and include the null entries again.
    idx = 0
    for i in range(sa.size):
        if sa.isnull(i):
            result[i] = -1
        else:
            tmp = indices[idx]
            tmp2 = non_na_index[tmp]
            result[i] = tmp2
            idx += 1

    return result


@numba.jit(nogil=True, nopython=True)
def _argsort_string_array_numpy(sa, indices, null_count):
    """
    Run numpy-like argsort on a pyarrow.StringArray.

    In contrast to Pandas, na-values will be included but sorted to the
    end.

    Parameters
    ----------
    sa: fletcher._numba_compat.NumbaStringArray
    indices: numpy.ndarray[int64]
    non_na_indices: numpy.ndarray[int64]
        Temporay array that holds the indices as they will be returned for
        Pandas, i.e. the index that would be retrieved by skipping the NA
        entries.
    """
    # Step 1: Intially fill indices and non_na_indices
    #   * indices will contain all indices for entries that are non-null, i.e.
    #     it has as many entries as the array has valid values.
    #   * non_na_indices will be will filled with the indices as if the null
    #     entries were omitted in the array. Null entries are represented as -1
    #     in this array.
    non_na_index = 0
    na_index = sa.size - null_count
    for i in range(sa.size):
        if sa.isnull(i):
            indices[na_index] = i
            na_index += 1
        else:
            indices[non_na_index] = i
            non_na_index += 1

    # Step 2: Recursively run quicksort, thereby sort indices but use
    #   the array for comparison.
    # _argquicksort_string_array(indices, sa, 0, non_na_index - 1)
    return indices


def argsort_string_array(chunked_array):
    """
    Sort a chunked array of type String in descending order.

    Parameters
    ----------
    chunked_array: pyarrow.ChunkedArray

    Returns
    -------
    sortindices: numpy.ndarry
    """
    if chunked_array.num_chunks != 1:
        raise NotImplementedError("Argsort only works on non-chunked data")
    array = chunked_array.chunk(0)
    sa = NumbaStringArray.make(array)
    indices = np.ones(len(array), dtype=int)
    return _argsort_string_array_numpy(sa, indices, array.null_count)
    return indices


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
