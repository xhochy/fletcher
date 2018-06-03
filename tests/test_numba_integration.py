# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import numba
import numpy as np
import pyarrow as pa
import pytest

# TODO: remove internal import
from fletcher._numba_compat import (
    NumbaStringArray,
    NumbaStringArrayBuilder,
    buffers_as_arrays,
)
from fletcher._algorithms import isnull, str_length, str_concat


@numba.jit(nogil=True, nopython=True)
def null_count(sa):
    result = 0

    for i in range(sa.size):
        result += sa.isnull(i)

    return result


@pytest.mark.parametrize(
    "array, expected",
    [
        (["foo", "bar", "baz"], 0),
        (["foo", "bar", None], 1),
        (["foo", None, "baz"], 1),
        ([None, "bar", "baz"], 1),
        (["foo", None, None], 2),
    ],
)
def test_null_count(array, expected):
    assert null_count(NumbaStringArray.make(array)) == expected


@pytest.mark.parametrize(
    "array, expected",
    [
        (["foo", "bar", "baz"], [False, False, False]),
        (["foo", "bar", None], [False, False, True]),
        (["foo", None, "baz"], [False, True, False]),
        ([None, "bar", "baz"], [True, False, False]),
        (["foo", None, None], [False, True, True]),
        (["föö", None], [False, True]),
    ],
)
@pytest.mark.parametrize("offset", [0, 1])
def test_isnull(array, expected, offset):
    array = pa.array(array, pa.string())[offset:]
    np.testing.assert_array_equal(
        isnull(NumbaStringArray.make(array)),
        np.asarray(expected[offset:], dtype=np.bool),
    )


@pytest.mark.parametrize(
    "array, expected",
    [
        (["f", "fo", "foo"], [1, 2, 3]),
        (["foo", "bar", None], [3, 3, 0]),
        (["foo", None, "baz"], [3, 0, 3]),
        ([None, "bar", "baz"], [0, 3, 3]),
        (["foo", None, None], [3, 0, 0]),
        ([None, None, None], [0, 0, 0]),
        pytest.mark.xfail(reason="non ascii not yet supported")((["föö"], [3])),
    ],
)
@pytest.mark.parametrize("offset", [0, 1])
def test_str_length(array, expected, offset):
    array = pa.array(array, pa.string())[offset:]
    np.testing.assert_array_equal(
        str_length(NumbaStringArray.make(array)),
        np.asarray(expected[offset:], dtype=np.int32),
    )


def test_str_concat():
    a1 = pa.array(["f", "ba", "baz"])
    a2 = pa.array(["oo", "r", ""])

    actual = str_concat(NumbaStringArray.make(a1), NumbaStringArray.make(a2))
    expected = NumbaStringArray.make(["foo", "bar", "baz"])

    np.testing.assert_array_equal(actual.missing, expected.missing)
    np.testing.assert_array_equal(actual.offsets, expected.offsets)
    np.testing.assert_array_equal(actual.data, expected.data)


def test_decode_example():
    strings = ["foo", "bar", "baz"]
    expected = strings[1].encode("utf32")
    expected = memoryview(expected)
    expected = np.asarray(expected).view(np.uint32)

    # remove endianness marker
    expected = expected.view(np.uint32)[1:]

    np.testing.assert_almost_equal(NumbaStringArray.make(strings).decode(1), expected)


@pytest.mark.parametrize(
    "data",
    [
        ["foo"],
        ["foo", None],
        [None, None, None, None],
        ["foo", "bar"],
        ["foo", "bar", "baz"],
    ],
)
def test_string_builder_simple(data):
    builder = NumbaStringArrayBuilder(2, 6)

    for s in data:
        if s is None:
            builder.finish_null()
            continue

        for c in s:
            builder.put_byte(ord(c))

        builder.finish_string()

    builder.finish()

    expected = pa.array(data, pa.string())
    missing, offsets, data = buffers_as_arrays(expected)

    np.testing.assert_array_equal(builder.missing, missing)
    np.testing.assert_array_equal(builder.offsets, offsets)
    np.testing.assert_array_equal(builder.data, data)
