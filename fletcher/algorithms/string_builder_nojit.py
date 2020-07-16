import math
from typing import List

import numba
import numpy as np
import pyarrow as pa


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
        self.buf.append(np.uint8(np.uint32(i32) & np.uint32(0xFF)))
        self.buf.append(np.uint8((np.uint32(i32) & np.uint32(0xFF00)) >> np.uint32(8)))
        self.buf.append(
            np.uint8((np.uint32(i32) & np.uint32(0xFF0000)) >> np.uint32(16))
        )
        self.buf.append(
            np.uint8((np.uint32(i32) & np.uint32(0xFF000000)) >> np.uint32(24))
        )

    def append_int16(self, i16):
        """Append a signed 16bit integer."""
        self.buf.append(np.uint8(i16 & np.uint16(0xFF)))
        self.buf.append(np.uint8((i16 & np.uint16(0xFF00)) >> np.uint16(8)))

    def append_int32(self, i32):
        """Append a signed 32bit integer."""
        self.append_uint32(i32)

    def append_int64(self, i64):
        """Append a signed 64bit integer."""
        self.buf.append(np.uint8(np.uint64(i64) & np.uint64(0xFF)))
        self.buf.append(np.uint8((np.uint64(i64) & np.uint64(0xFF00)) >> np.uint64(8)))
        self.buf.append(
            np.uint8((np.uint64(i64) & np.uint64(0xFF0000)) >> np.uint64(16))
        )
        self.buf.append(
            np.uint8((np.uint64(i64) & np.uint64(0xFF000000)) >> np.uint64(24))
        )
        self.buf.append(
            np.uint8((np.uint64(i64) & np.uint64(0xFF00000000)) >> np.uint64(32))
        )
        self.buf.append(
            np.uint8((np.uint64(i64) & np.uint64(0xFF0000000000)) >> np.uint64(40))
        )
        self.buf.append(
            np.uint8((np.uint64(i64) & np.uint64(0xFF000000000000)) >> np.uint64(48))
        )
        self.buf.append(
            np.uint8((np.uint64(i64) & np.uint64(0xFF00000000000000)) >> np.uint64(56))
        )

    def append_bytes(self, ptr, length):
        """Append a range of bytes."""
        for i in range(length):
            self.buf.append(np.uint8(ptr[i]))

    def get_uint8(self, idx):
        return np.uint8(self.buf[idx])

    def get_int16(self, idx):
        return np.int16(
            np.uint16(self.buf[idx * 2]) | (np.uint16(self.buf[idx * 2 + 1]) << 8)
        )

    def get_int32(self, idx):
        return np.int32(
            np.uint32(self.buf[idx * 4])
            | (np.uint32(self.buf[idx * 4 + 1]) << 8)
            | (np.uint32(self.buf[idx * 4 + 2]) << 16)
            | (np.uint32(self.buf[idx * 4 + 3]) << 24)
        )

    def get_uint32(self, idx):
        return (
            np.uint32(self.buf[idx * 4])
            | (np.uint32(self.buf[idx * 4 + 1]) << 8)
            | (np.uint32(self.buf[idx * 4 + 2]) << 16)
            | (np.uint32(self.buf[idx * 4 + 3]) << 24)
        )

    def get_int64(self, idx):
        return np.int64(
            np.uint64(self.buf[idx * 8])
            | (np.uint64(self.buf[idx * 8 + 1]) << np.uint64(8))
            | (np.uint64(self.buf[idx * 8 + 2]) << np.uint64(16))
            | (np.uint64(self.buf[idx * 8 + 3]) << np.uint64(24))
            | (np.uint64(self.buf[idx * 8 + 4]) << np.uint64(32))
            | (np.uint64(self.buf[idx * 8 + 5]) << np.uint64(40))
            | (np.uint64(self.buf[idx * 8 + 6]) << np.uint64(48))
            | (np.uint64(self.buf[idx * 8 + 7]) << np.uint64(56))
        )

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
        self.buf = []  # type: List[np.uint8]
        self.size = 0

    def append_true(self):
        if self.size % 8 == 0:
            self.buf.append(np.uint8(0))
        self.buf[-1] |= 1 << (self.size % 8)
        self.size += 1

    def append_false(self):
        if self.size % 8 == 0:
            self.buf.append(np.uint8(0))
        self.size += 1

    def get(self, idx):
        byte_offset = idx // 8
        bit_offset = idx % 8
        mask = np.uint8(1 << bit_offset)
        return self.buf[byte_offset] & mask != 0

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
    valid_bits = pa.py_buffer(np.array(sba.valid_bits.buf, dtype=np.uint8))
    value_offsets = np.array(
        sba.value_offsets.buf[: len(sba.value_offsets.buf)], dtype=np.uint8
    )
    value_offsets = pa.py_buffer(value_offsets)
    data = pa.py_buffer(np.copy(sba.data.buf[: len(sba.data.buf)]))
    sba.delete()
    return pa.Array.from_buffers(
        typ, sba.length, [valid_bits, value_offsets, data], sba.null_count
    )
