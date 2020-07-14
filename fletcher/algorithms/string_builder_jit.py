import math

import numba
import numpy as np
import pyarrow as pa
from cffi import FFI

ffi = FFI()
ffi.cdef("void free(void* ptr);")
ffi.cdef("void* malloc(size_t size);")
ffi.cdef("void* memcpy( void *dest, const void *src, size_t count );")
libc = ffi.dlopen(None)
malloc = libc.malloc
free = libc.free
memcpy = libc.memcpy


@numba.jitclass(
    [
        ("ptr", numba.types.voidptr),
        ("capacity", numba.int64),
        ("size", numba.int64),
        ("buf", numba.byte[:]),
    ]
)
class ByteVector:
    """
    Builder that constructs a buffer based on byte-sized chunks.

    As the memory is owned by this object but we cannot override __del__,
    you need to explicitly call delete() to free the native memory.
    """

    def __init__(self, initial_size: int):
        self.ptr = malloc(initial_size)
        self.capacity = max(initial_size, 5)
        self.buf = numba.carray(self.ptr, self.capacity, numba.byte)
        self.size = 0

    def delete(self):
        free(self.ptr)

    def append(self, byte):
        """Append a single byte to the stream."""
        if self.size + 1 > self.capacity:
            self.expand()
        self.buf[self.size] = byte
        self.size += 1

    def append_uint32(self, i32):
        """Append an unsigned 32bit integer."""
        if self.size + 4 > self.capacity:
            self.expand()
        self.buf[self.size] = i32 & 0xFF
        self.buf[self.size + 1] = (i32 & 0xFF00) >> 8
        self.buf[self.size + 2] = (i32 & 0xFF0000) >> 16
        self.buf[self.size + 3] = (i32 & 0xFF000000) >> 24
        self.size += 4

    def append_int16(self, i16):
        """Append a signed 16bit integer."""
        if self.size + 2 > self.capacity:
            self.expand()
        self.buf[self.size] = i16 & 0xFF
        self.buf[self.size + 1] = (i16 & 0xFF00) >> 8
        self.size += 2

    def append_int32(self, i32):
        """Append a signed 32bit integer."""
        if self.size + 4 > self.capacity:
            self.expand()
        self.buf[self.size] = i32 & 0xFF
        self.buf[self.size + 1] = (i32 & 0xFF00) >> 8
        self.buf[self.size + 2] = (i32 & 0xFF0000) >> 16
        self.buf[self.size + 3] = (i32 & 0xFF000000) >> 24
        self.size += 4

    def append_int64(self, i64):
        """Append a signed 64bit integer."""
        if self.size + 8 > self.capacity:
            self.expand()
        self.buf[self.size] = i64 & 0xFF
        self.buf[self.size + 1] = (i64 & 0xFF00) >> 8
        self.buf[self.size + 2] = (i64 & 0xFF0000) >> 16
        self.buf[self.size + 3] = (i64 & 0xFF000000) >> 24
        self.buf[self.size + 4] = (i64 & 0xFF00000000) >> 32
        self.buf[self.size + 5] = (i64 & 0xFF0000000000) >> 40
        self.buf[self.size + 6] = (i64 & 0xFF000000000000) >> 48
        self.buf[self.size + 7] = (i64 & 0xFF00000000000000) >> 56
        self.size += 8

    def append_bytes(self, ptr, length):
        """Append a range of bytes."""
        while self.size + length > self.capacity:
            self.expand()
        for i in range(length):
            self.buf[self.size] = ptr[i]
            self.size += 1

    def expand(self):
        """
        Double the size of the underlying buffer and copy over the existing data.

        This allocates a new buffer and copies the data.
        """
        new_ptr = malloc(2 * self.capacity)
        memcpy(new_ptr, self.ptr, self.capacity)
        self.capacity = 2 * self.capacity
        free(self.ptr)
        self.ptr = new_ptr
        self.buf = numba.carray(self.ptr, self.capacity, numba.byte)


@numba.jitclass(
    [
        ("ptr", numba.types.voidptr),
        ("capacity", numba.int64),
        ("size", numba.int64),
        ("buf", numba.byte[:]),
    ]
)
class BitVector:
    """
    Builder that constructs a buffer based on bit-packed chunks.

    As the memory is owned by this object but we cannot override __del__,
    you need to explicitly call delete() to free the native memory.
    """

    def __init__(self, initial_size: int):
        self.ptr = malloc(initial_size)
        self.capacity = max(initial_size, 5)
        self.buf = numba.carray(self.ptr, self.capacity, numba.byte)
        self.size = 0

    def append_true(self):
        if self.size + 1 > (8 * self.capacity):
            self.expand()

        byte_offset = self.size // 8
        bit_offset = self.size % 8
        mask = np.uint8(1 << bit_offset)
        self.buf[byte_offset] = self.buf[byte_offset] | mask

        self.size += 1

    def append_false(self):
        if self.size + 1 > (8 * self.capacity):
            self.expand()

        byte_offset = self.size // 8
        bit_offset = self.size % 8
        mask = np.uint8(1 << bit_offset)
        self.buf[byte_offset] = self.buf[byte_offset] & (~mask)

        self.size += 1

    def delete(self):
        free(self.ptr)

    def expand(self):
        """
        Double the size of the underlying buffer and copy over the existing data.

        This allocates a new buffer and copies the data.
        """
        new_ptr = malloc(2 * self.capacity)
        memcpy(new_ptr, self.ptr, self.capacity)
        self.capacity = 2 * self.capacity
        free(self.ptr)
        self.ptr = new_ptr
        self.buf = numba.carray(self.ptr, self.capacity, numba.byte)
        print("Cap: ", {self.capacity})


@numba.jit
def byte_for_bits(num_bits):
    # We need to use math as numpy would return a float: https://github.com/numpy/numpy/issues/9068
    return math.ceil(num_bits / 8)


def bit_vector_to_pa_boolarray(bv: BitVector) -> pa.BooleanArray:
    bools = pa.py_buffer(np.copy(bv.buf[: byte_for_bits(bv.size)]))
    return pa.BooleanArray.from_buffers(
        pa.bool_(), bv.size, [None, bools], null_count=0
    )


@numba.jitclass(
    [
        ("length", numba.uint64),
        ("null_count", numba.uint64),
        ("current_offset", numba.uint64),
        ("valid_bits", BitVector.class_type.instance_type),  # type: ignore
        ("value_offsets", ByteVector.class_type.instance_type),  # type: ignore
        ("data", ByteVector.class_type.instance_type),  # type: ignore
    ]
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
        np.copy(sba.valid_bits.buf[: byte_for_bits(sba.valid_bits.size)])
    )
    value_offsets = np.copy(sba.value_offsets.buf[: sba.value_offsets.size])
    value_offsets = pa.py_buffer(value_offsets)
    data = pa.py_buffer(np.copy(sba.data.buf[: sba.data.size]))
    sba.delete()
    return pa.Array.from_buffers(
        typ, sba.length, [valid_bits, value_offsets, data], sba.null_count
    )
