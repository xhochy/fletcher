import math
from typing import List

import numba
import numpy as np
import pyarrow as pa
from cffi import FFI

class ByteVector:
    """
    Builder that constructs a buffer based on byte-sized chunks.

    As the memory is owned by this object but we cannot override __del__,
    you need to explicitly call delete() to free the native memory.
    """

    def __init__(self, initial_size: int):
        self.buf = []  # type: List[numba.byte]

    def delete(self):
        pass

    def append(self, byte):
        """Append a single byte to the stream."""
        self.buf.append(byte)

    def append_uint32(self, i32):
        """Append an unsigned 32bit integer."""
        self.buf.append(numba.byte(i32 & 0xFF))
        self.buf.append(numba.byte(i32 & 0xFF00) >> 8)
        self.buf.append(numba.byte(i32 & 0xFF0000) >> 16)
        self.buf.append(numba.byte(i32 & 0xFF000000) >> 24)

    def append_int16(self, i16):
        """Append a signed 16bit integer."""
        self.buf.append(numba.byte(i16 & 0xFF))
        self.buf.append(numba.byte(i16 & 0xFF00) >> 8)

    def append_int32(self, i32):
        """Append a signed 32bit integer."""
        self.append_uint32(i32)

    def append_int64(self, i64):
        """Append a signed 64bit integer."""
        self.buf.append(numba.byte(i64 & 0xFF))
        self.buf.append(numba.byte(i64 & 0xFF00) >> 8)
        self.buf.append(numba.byte(i64 & 0xFF0000) >> 16)
        self.buf.append(numba.byte(i64 & 0xFF000000) >> 24)
        self.buf.append(numba.byte(i64 & 0xFF00000000) >> 32)
        self.buf.append(numba.byte(i64 & 0xFF0000000000) >> 40)
        self.buf.append(numba.byte(i64 & 0xFF000000000000) >> 48)
        self.buf.append(numba.byte(i64 & 0xFF00000000000000) >> 56)

    def append_bytes(self, ptr, length):
        """Append a range of bytes."""
        for i in range(length):
            self.buf.append(numba.byte(ptr[i]))

    def expand(self):
        """
        Double the size of the underlying buffer and copy over the existing data.

        This allocates a new buffer and copies the data.
        """
        pass


class BitVector:
    """
    Builder that constructs a buffer based on bit-packed chunks.

    As the memory is owned by this object but we cannot override __del__,
    you need to explicitly call delete() to free the native memory.
    """

    def __init__(self, initial_size: int):
        self.buf = []  # type: List[numba.byte]
        self.size = 0

    def append_true(self):
        if self.size % 8 == 0:
            self.buf.append(numba.byte(0))
        self.buf[-1] |= 1 << (self.size % 8)
        self.size += 1

    def append_false(self):
        if self.size % 8 == 0:
            self.buf.append(numba.byte(0))
        self.size += 1

    def delete(self):
        pass

    def expand(self):
        """
        Double the size of the underlying buffer and copy over the existing data.

        This allocates a new buffer and copies the data.
        """
        pass


def byte_for_bits(num_bits):
    # We need to use math as numpy would return a float: https://github.com/numpy/numpy/issues/9068
    return math.ceil(num_bits / 8)


def bit_vector_to_pa_boolarray(bv: BitVector) -> pa.BooleanArray:
    bools = pa.py_buffer(np.copy(bv.buf[: byte_for_bits(len(bv.buf))]))
    return pa.BooleanArray.from_buffers(
        pa.bool_(), len(bv.buf), [None, bools], null_count=0
    )


class StringArrayBuilder:
    """
    Numba-based builder to construct pyarrow.StringArray instances.

    As Numba doesn't allow us to override __del__, we must always call delete
    to free up the used (native) memory.
    """

    def __init__(self, initial_size: int):
        self.length = 0
        self.null_count = 0
        self.current_offset = 0

        self.valid_bits = BitVector(byte_for_bits(initial_size))
        self.value_offsets = ByteVector(initial_size)
        self.value_offsets.append_uint32(0)
        self.data = ByteVector(initial_size)

    def append_null(self):
        self.valid_bits.append_false()
        self.value_offsets.append_uint32(self.current_offset)
        self.length += 1
        self.null_count += 1

    def append_value(self, ptr, length):
        self.valid_bits.append_true()
        self.length += 1
        self.current_offset += length
        self.data.append_bytes(ptr, length)
        self.value_offsets.append_uint32(self.current_offset)

    def delete(self):
        # Free all resources
        self.valid_bits.delete()
        self.value_offsets.delete()
        self.data.delete()


def finalize_string_array(sba, typ) -> pa.Array:
    """
    Take a StringArrayBuilder and returns a pyarrow.StringArray.
    The native memory in the StringArrayBuilder is free'd during this method
    call and this is unusable afterwards but also doesn't leak any memory.
    """
    # TODO: Can we handle this without a copy? Currently there is no way
    # to pass a custom destructor any pyarrow.*_buffer function.
    valid_bits = pa.py_buffer(
        np.copy(sba.valid_bits.buf[: byte_for_bits(len(sba.valid_bits.buf))])
    )
    value_offsets = np.array(sba.value_offsets.buf[: len(sba.value_offsets.buf)], dtype=np.uint8)
    value_offsets = pa.py_buffer(value_offsets)
    data = pa.py_buffer(np.copy(sba.data.buf[: len(sba.data.buf)]))
    sba.delete()
    return pa.Array.from_buffers(
        typ, sba.length, [valid_bits, value_offsets, data], sba.null_count
    )
