from datetime import timedelta
from typing import Any, List, Tuple

import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pandas as pd
import pyarrow as pa
import pytest
from hypothesis import given, settings

import fletcher.algorithms.string_builder as sb1
import fletcher.algorithms.string_builder_nojit as sb2
from fletcher._algorithms import (
    _extract_data_buffer_as_np_array,
    _merge_valid_bitmaps,
    max_op,
    min_op,
    np_ufunc_op,
    prod_op,
    sum_op,
)
from fletcher.algorithms.utils.chunking import (
    _calculate_chunk_offsets,
    _combined_in_chunk_offsets,
    _in_chunk_offsets,
)


def _is_na(a):
    return (a is pa.NA) or (a is None) or (np.isnan(a))


def assert_allclose_na(a, b):
    """assert_allclose with a broader NA/nan/None definition."""
    if _is_na(a) and _is_na(b):
        pass
    else:
        npt.assert_allclose(a, b)


@pytest.mark.parametrize(
    "op, pandas_op", [(sum_op, pd.Series.sum), (prod_op, pd.Series.prod)]
)
@settings(deadline=timedelta(milliseconds=1000))
@given(
    data=st.lists(st.one_of(st.floats(max_value=10.0, min_value=-10), st.none())),
    skipna=st.booleans(),
)
def test_reduce_op(data, skipna, op, pandas_op):
    arrow = pa.array(data, type=pa.float64(), from_pandas=True)
    pandas = pd.Series(data, dtype=float)

    assert_allclose_na(op(arrow, skipna), pandas_op(pandas, skipna=skipna))

    # Split in the middle and check whether this still works
    if len(data) > 2:
        arrow = pa.chunked_array(
            [
                pa.array(data[: len(data) // 2], type=pa.float64(), from_pandas=True),
                pa.array(data[len(data) // 2 :], type=pa.float64(), from_pandas=True),
            ]
        )
        assert_allclose_na(op(arrow, skipna), pandas_op(pandas, skipna=skipna))


@pytest.mark.parametrize(
    "op, pandas_op", [(min_op, pd.Series.min), (max_op, pd.Series.max)]
)
@settings(deadline=timedelta(milliseconds=1000))
@given(
    data=st.lists(st.one_of(st.floats(max_value=10.0), st.none())), skipna=st.booleans()
)
def test_reduce_op_no_identity(data, skipna, op, pandas_op):
    arrow = pa.array(data, type=pa.float64(), from_pandas=True)
    pandas = pd.Series(data, dtype=float)
    should_raise = arrow.null_count == len(arrow) and (skipna or len(arrow) == 0)

    if should_raise:
        with pytest.raises(ValueError):
            assert_allclose_na(op(arrow, skipna), pandas_op(pandas, skipna=skipna))
    else:
        assert_allclose_na(op(arrow, skipna), pandas_op(pandas, skipna=skipna))

    # Split in the middle and check whether this still works
    if len(data) > 2:
        arrow = pa.chunked_array(
            [
                pa.array(data[: len(data) // 2], type=pa.float64(), from_pandas=True),
                pa.array(data[len(data) // 2 :], type=pa.float64(), from_pandas=True),
            ]
        )
        if should_raise:
            with pytest.raises(ValueError):
                assert_allclose_na(op(arrow, skipna), pandas_op(pandas, skipna=skipna))
        else:
            assert_allclose_na(op(arrow, skipna), pandas_op(pandas, skipna=skipna))


def test_calculate_chunk_offsets():
    arr = pa.chunked_array([[1, 1, 1]])
    npt.assert_array_equal(_calculate_chunk_offsets(arr), np.array([0]))
    arr = pa.chunked_array([[1], [1, 1]])
    npt.assert_array_equal(_calculate_chunk_offsets(arr), np.array([0, 1]))
    arr = pa.chunked_array([[1, 1], [1]])
    npt.assert_array_equal(_calculate_chunk_offsets(arr), np.array([0, 2]))


def check_valid_in_offsets(
    arr: pa.ChunkedArray, in_offsets: List[Tuple[int, int, int]]
) -> None:
    if arr.num_chunks == 0:
        assert in_offsets == []
        return

    # We always start at the beginning
    assert in_offsets[0][0] == 0
    assert in_offsets[0][1] == 0

    # Overall, the chunk offsets must have the same length as the array
    assert sum(x[2] for x in in_offsets) == len(arr)


@given(data=st.lists(st.lists(st.integers(min_value=0, max_value=10))))
def test_in_chunk_offsets(data: List[List[int]]):
    arr = pa.chunked_array(data, type=pa.int64())
    # Simple case: Passing in the actual chunk offsets should yield a valid selection
    offsets = list(_calculate_chunk_offsets(arr))
    in_offsets = _in_chunk_offsets(arr, offsets)
    check_valid_in_offsets(arr, in_offsets)


def test_combined_in_chunk_offsets():
    a = pa.chunked_array([[]])
    b = pa.chunked_array([[]])
    in_a_offsets, in_b_offsets = _combined_in_chunk_offsets(a, b)
    assert in_a_offsets == [(0, 0, 0)]
    assert in_b_offsets == [(0, 0, 0)]

    a = pa.chunked_array([[1]])
    b = pa.chunked_array([[2]])
    in_a_offsets, in_b_offsets = _combined_in_chunk_offsets(a, b)
    assert in_a_offsets == [(0, 0, 1)]
    assert in_b_offsets == [(0, 0, 1)]

    a = pa.chunked_array([[1, 2], [3, 4, 5]])
    b = pa.chunked_array([[1], [2, 3], [4, 5]])
    in_a_offsets, in_b_offsets = _combined_in_chunk_offsets(a, b)
    assert in_a_offsets == [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 2)]
    assert in_b_offsets == [(0, 0, 1), (1, 0, 1), (1, 1, 1), (2, 0, 2)]


@pytest.mark.parametrize("data", [[1, 2, 4, 5], [1.0, 0.5, 4.0, 5.0]])
def test_extract_data_buffer_as_np_array(data):
    arr = pa.array(data)
    result = _extract_data_buffer_as_np_array(arr)
    expected = np.array(data)
    npt.assert_array_equal(result, expected)
    result = _extract_data_buffer_as_np_array(arr[2:4])
    expected = np.array(data[2:4])
    npt.assert_array_equal(result, expected)


def assert_content_equals_array(result, expected):
    """Assert that the result is an Arrow structure and the content matches an array."""
    assert isinstance(result, (pa.Array, pa.ChunkedArray))
    if isinstance(result, pa.ChunkedArray):
        result = pa.concat_arrays(result.iterchunks())
    assert result.equals(expected)


def check_np_ufunc(a, b, expected):
    result = np_ufunc_op(a, b, np.ndarray.__add__)
    assert_content_equals_array(result, expected)
    result = np_ufunc_op(b, a, np.ndarray.__add__)
    assert_content_equals_array(result, expected)


def test_np_ufunc_op_chunked_chunked():
    a = pa.chunked_array([[1, 2], [3, None, None]])
    b = pa.chunked_array([[1], [2, 3], [4, None]])
    expected = pa.array([2, 4, 6, None, None])
    check_np_ufunc(a, b, expected)


def test_np_ufunc_op_chunked_flat():
    a = pa.chunked_array([[1, 2], [3, None, None]])
    b = pa.array([1, 2, 3, 4, None])
    expected = pa.array([2, 4, 6, None, None])
    check_np_ufunc(a, b, expected)


def test_np_ufunc_op_chunked_np_array():
    a = pa.chunked_array([[1, 2], [3, None]])
    b = np.array([1, 2, 3, 4])
    expected = pa.array([2, 4, 6, None])
    check_np_ufunc(a, b, expected)


def test_np_ufunc_op_chunked_scalar():
    a = pa.chunked_array([[1, 2], [3, None]])
    b = 4
    expected = pa.array([5, 6, 7, None])
    check_np_ufunc(a, b, expected)


def test_np_ufunc_op_flat_flat():
    a = pa.array([1, 2, 3, None, None])
    b = pa.array([1, 2, 3, 4, None])
    expected = pa.array([2, 4, 6, None, None])
    check_np_ufunc(a, b, expected)


def test_np_ufunc_op_flat_np_array():
    a = pa.array([1, 2, 3, None])
    b = np.array([1, 2, 3, 4])
    expected = pa.array([2, 4, 6, None])
    check_np_ufunc(a, b, expected)


def test_np_ufunc_op_flat_scalar():
    a = pa.array([1, 2, 3, None])
    b = 4
    expected = pa.array([5, 6, 7, None])
    check_np_ufunc(a, b, expected)


def test_merge_valid_bitmaps():
    a = pa.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    b = pa.array([1, 1, 1, None, None, None, 1, 1, 1])

    expected = np.array([0xFF, 0x1], dtype=np.uint8)
    result = _merge_valid_bitmaps(a, a)
    npt.assert_array_equal(result, expected)
    expected = np.array([0xC7, 0x1], dtype=np.uint8)
    result = _merge_valid_bitmaps(a, b)
    npt.assert_array_equal(result, expected)

    expected = np.array([0x1], dtype=np.uint8)
    result = _merge_valid_bitmaps(a.slice(8, 1), a.slice(8, 1))
    npt.assert_array_equal(result, expected)

    expected = np.array([0xF], dtype=np.uint8)
    result = _merge_valid_bitmaps(a.slice(0, 4), a.slice(0, 4))
    npt.assert_array_equal(result, expected)
    expected = np.array([0x7], dtype=np.uint8)
    result = _merge_valid_bitmaps(a.slice(0, 4), b.slice(0, 4))
    npt.assert_array_equal(result, expected)

    expected = np.array([0xF], dtype=np.uint8)
    result = _merge_valid_bitmaps(a.slice(5, 4), a.slice(5, 4))
    npt.assert_array_equal(result, expected)
    expected = np.array([0xE], dtype=np.uint8)
    result = _merge_valid_bitmaps(a.slice(5, 4), b.slice(5, 4))
    npt.assert_array_equal(result, expected)

    expected = np.array([0x3], dtype=np.uint8)
    result = _merge_valid_bitmaps(a.slice(5, 2), a.slice(5, 2))
    npt.assert_array_equal(result, expected)
    expected = np.array([0x2], dtype=np.uint8)
    result = _merge_valid_bitmaps(a.slice(5, 2), b.slice(5, 2))
    npt.assert_array_equal(result, expected)

    expected = np.array([0x3], dtype=np.uint8)
    result = _merge_valid_bitmaps(a.slice(5, 2), a.slice(3, 2))
    npt.assert_array_equal(result, expected)
    expected = np.array([0x0], dtype=np.uint8)
    result = _merge_valid_bitmaps(a.slice(5, 2), b.slice(3, 2))
    npt.assert_array_equal(result, expected)


@pytest.fixture(params=["jit", "nojit"], scope="session")
def string_builder_variant(request):
    """Whether to test the chunked or continuous implementation."""
    return request.param


@pytest.fixture(
    params=[
        [
            "000000000000000000000000000000000000000000İࠀࠀࠀࠀ𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐤱000000000000𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀"
        ],
        ["\x80"],
        [],
        [""],
        [None],
        ["a"],
        ["aa"],
        ["", None],
        ["a", None],
        [None, ""],
        [None, "a"],
        ["", "", "", "", None, None, None, None, ""],
    ],
    scope="session",
)
def string_builder_test_data(request):
    """Explicit test data for test_stringbuilder."""
    return request.param


def test_stringbuilder(string_builder_variant, string_builder_test_data):
    _stringbuilder_test_(
        string_builder_test_data,
        pa.array(string_builder_test_data),
        string_builder_variant,
    )


@settings(deadline=None)
@given(data=st.lists(st.one_of(st.text(), st.none())))
def test_stringbuilder_auto(string_builder_variant, data):
    _stringbuilder_test_(data, pa.array(data), string_builder_variant)


def _stringbuilder_test_(values, expected, string_builder_variant):
    if string_builder_variant == "nojit":
        sb = sb2  # type: Any
    else:
        sb = sb1
    builder = sb.StringArrayBuilder(0)
    for value in values:
        if value is None:
            builder.append_null()
        else:
            encoded_value = bytes(value, encoding="utf-8")
            builder.append_value(encoded_value, len(encoded_value))
    result = sb.finalize_string_array(builder, pa.string())
    npt.assert_array_equal(result, expected)


@settings(deadline=None)
@given(data=st.lists(st.integers(-2 ** 31, 2 ** 31 - 1)))
def test_vector_auto(string_builder_variant, data):
    if string_builder_variant == "nojit":
        sb = sb2  # type: Any
    else:
        sb = sb1
    vec = sb.ByteVector(0)
    for num in data:
        vec.append_int32(np.int32(num))
    for idx in range(len(data)):
        assert vec.get_int32(idx) == np.int32(data[idx])
