import numba
import numpy as np
import pyarrow as pa

_string_buffer_types = np.uint8, np.uint32, np.uint8


def _buffers_as_arrays(sa):
    buffers = sa.buffers()
    return tuple(np.asarray(b).view(t) for b, t in zip(buffers, _string_buffer_types))


@numba.jitclass([
    ('missing', numba.uint8[:]),
    ('offsets', numba.uint32[:]),
    ('data', numba.uint8[:]),
])
class NumbaStringArray:
    """Wrapper around arrow's StringArray for use in numba functions.

    Usage::

        NumbaStringArray.make(array)
    """
    def __init__(self, missing, offsets, data):
        self.missing = missing
        self.offsets = offsets
        self.data = data

    @property
    def byte_size(self):
        return self.data.shape[0]

    @property
    def size(self):
        return len(self.offsets) - 1

    def isnull(self, str_idx):
        byte_idx = str_idx // 8
        bit_mask = 1 << (str_idx % 8)
        return (self.missing[byte_idx] & bit_mask) == 0

    def byte_length(self, str_idx):
        return self.offsets[str_idx + 1] - self.offsets[str_idx]

    def get_byte(self, str_idx, byte_idx):
        full_idx = self.offsets[str_idx] + byte_idx
        return self.data[full_idx]

    def length(self, str_idx):
        result = 0
        byte_length = self.byte_length(str_idx)
        current = 0

        while current < byte_length:
            _, inc = self.get(str_idx, current)
            current += inc
            result += 1

        return result

    # TODO: implement this
    def get(self, str_idx, byte_idx):
        b = self.get_byte(str_idx, byte_idx)
        if b > 127:
            raise ValueError()

        return b, 1

    def decode(self, str_idx):

        byte_length = self.byte_length(str_idx)
        buffer = np.zeros(byte_length, np.int32)

        i = 0
        j = 0
        while i < byte_length:
            code, inc = self.get(str_idx, i)
            buffer[j] = code

            i += inc
            j += 1

        return buffer[:j]


def _make(sa):
    if not isinstance(sa, pa.StringArray):
        sa = pa.array(sa)

    return NumbaStringArray(*_buffers_as_arrays(sa))


# @classmethod does not seem to be supported
NumbaStringArray.make = _make
