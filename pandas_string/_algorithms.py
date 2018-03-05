import numba
import numpy as np

from ._numba_compat import NumbaStringArray


@numba.jit(nogil=True, nopython=True)
def is_null(sa):
    result = np.empty(sa.size, np.uint8)

    for i in range(sa.size):
        result[i] = sa.isnull(i)

    return result


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

    return NumbaStringArray(result_missing, result_offsets, result_data)
