# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import math

import numba
import numpy as np
import pyarrow as pa
import types
import six


@numba.jitclass(
    [
        ("missing", numba.uint8[:]),
        ("offsets", numba.uint32[:]),
        ("data", numba.optional(numba.uint8[:])),
        ("offset", numba.int64),
    ]
)
class NumbaStringArray(object):
    """Wrapper around arrow's StringArray for use in numba functions.

    Usage::

        NumbaStringArray.make(array)
    """

    def __init__(self, missing, offsets, data, offset):
        self.missing = missing
        self.offsets = offsets
        self.data = data
        self.offset = offset

    @property
    def byte_size(self):
        # TODO: offset?
        return self.data.shape[0]

    @property
    def size(self):
        return len(self.offsets) - 1 - self.offset

    def isnull(self, str_idx):
        str_idx += self.offset
        byte_idx = str_idx // 8
        bit_mask = 1 << (str_idx % 8)
        return (self.missing[byte_idx] & bit_mask) == 0

    def elements_lt(self, left, right):
        """
        Check whether one non-null entry is lower than the other.
        """
        left_length = self.byte_length(left)
        right_length = self.byte_length(right)
        common_length = min(left_length, right_length)

        left_offset = self.offsets[self.offset + left]
        right_offset = self.offsets[self.offset + right]
        for i in range(common_length):
            left_char = self.data[left_offset + i]
            right_char = self.data[right_offset + i]
            if left_char != right_char:
                return left_char < right_char
        return left_length < right_length

    def byte_length(self, str_idx):
        str_idx += self.offset
        return self.offsets[str_idx + 1] - self.offsets[str_idx]

    def get_byte(self, str_idx, byte_idx):
        str_idx += self.offset
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


def _make(cls, sa):
    if not isinstance(sa, pa.StringArray):
        sa = pa.array(sa, pa.string())

    buffers = sa.buffers()

    if buffers[0] is None:
        missing = np.ones((len(sa) + 7) // 8, dtype=np.uint8) * 255
    else:
        missing = np.asarray(buffers[0]).view(np.uint8)
    offsets = np.asarray(buffers[1]).view(np.uint32)
    if buffers[2] is None:
        data = np.ones(0, dtype=np.uint8)
    else:
        data = np.asarray(buffers[2]).view(np.uint8)

    return cls(missing, offsets, data, offset=sa.offset)


# @classmethod does not seem to be supported
NumbaStringArray.make = types.MethodType(_make, NumbaStringArray)


@numba.jitclass(
    [("start", numba.uint32), ("end", numba.uint32), ("data", numba.uint8[:])]
)
class NumbaString(object):

    def __init__(self, data, start=0, end=None):
        if end is None:
            end = data.shape[0]

        self.data = data
        self.start = start
        self.end = end

    @property
    def length(self):
        return self.end - self.start

    def get_byte(self, i):
        return self.data[self.start + i]


def _make_string(cls, obj):
    if isinstance(obj, six.text_type):
        data = obj.encode("utf8")
        data = np.asarray(memoryview(data))

        return cls(data, 0, len(data))

    raise TypeError()


NumbaString.make = types.MethodType(_make_string, NumbaString)


@numba.jitclass(
    [
        ("missing", numba.uint8[:]),
        ("offsets", numba.uint32[:]),
        ("data", numba.optional(numba.uint8[:])),
        ("string_position", numba.uint32),
        ("byte_position", numba.uint32),
        ("string_capacity", numba.uint32),
        ("byte_capacity", numba.uint32),
    ]
)
class NumbaStringArrayBuilder(object):

    def __init__(self, string_capacity, byte_capacity):
        self.missing = np.ones(_missing_capactiy(string_capacity), np.uint8)
        self.offsets = np.zeros(string_capacity + 1, np.uint32)
        self.data = np.zeros(byte_capacity, np.uint8)
        self.string_position = 0
        self.byte_position = 0

        self.string_capacity = string_capacity
        self.byte_capacity = byte_capacity

    def increase_string_capacity(self, string_capacity):
        assert string_capacity > self.string_capacity

        missing = np.zeros(_missing_capactiy(string_capacity), np.uint8)
        missing[: _missing_capactiy(self.string_capacity)] = self.missing
        self.missing = missing

        offsets = np.zeros(string_capacity + 1, np.uint32)
        offsets[: self.string_capacity + 1] = self.offsets
        self.offsets = offsets

        self.string_capacity = string_capacity

    def increase_byte_capacity(self, byte_capacity):
        assert byte_capacity > self.byte_capacity

        data = np.zeros(byte_capacity, np.uint8)
        data[: self.byte_capacity] = self.data
        self.data = data

        self.byte_capacity = byte_capacity

    def put_byte(self, b):
        if self.byte_position >= self.byte_capacity:
            self.increase_byte_capacity(int(math.ceil(1.2 * self.byte_capacity)))

        self.data[self.byte_position] = b
        self.byte_position += 1

    def finish_string(self):
        if self.string_position >= self.string_capacity:
            self.increase_string_capacity(int(math.ceil(1.2 * self.string_capacity)))

        self.offsets[self.string_position + 1] = self.byte_position

        byte_idx = self.string_position // 8
        self.missing[byte_idx] |= 1 << (self.string_position % 8)

        self.string_position += 1

    def finish_null(self):
        if self.string_position >= self.string_capacity:
            self.increase_string_capacity(int(math.ceil(1.2 * self.string_capacity)))

        self.offsets[self.string_position + 1] = self.byte_position

        byte_idx = self.string_position // 8
        self.missing[byte_idx] &= ~(1 << (self.string_position % 8))

        self.string_position += 1

    def finish(self):
        self.missing = self.missing[: _missing_capactiy(self.string_position)]
        self.offsets = self.offsets[: self.string_position + 1]
        self.data = self.data[: self.byte_position]


@numba.jit
def _missing_capactiy(capacity):
    return int(math.ceil(capacity / 8))
