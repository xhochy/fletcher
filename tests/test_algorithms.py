from datetime import timedelta

import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pandas as pd
import pyarrow as pa
import pytest
from hypothesis import example, given, settings

from fletcher._algorithms import (
    _calculate_chunk_offsets,
    _combined_in_chunk_offsets,
    _extract_data_buffer_as_np_array,
    all_op,
    any_op,
    max_op,
    min_op,
    np_ufunc_op,
    prod_op,
    sum_op,
)


@settings(deadline=timedelta(milliseconds=1000))
@given(data=st.lists(st.one_of(st.booleans(), st.none())), skipna=st.booleans())
@example([], False)
@example([], True)
# Test with numpy.array as input.
# This has the caveat that the missing buffer is None.
@example(np.ones(10).astype(bool), False)
@example(np.ones(10).astype(bool), True)
def test_any_op(data, skipna):
    arrow = pa.array(data, type=pa.bool_())
    # TODO(pandas-0.26): Use pandas.BooleanArray
    # https://github.com/pandas-dev/pandas/issues/27709 / https://github.com/pandas-dev/pandas/issues/12863
    pandas = pd.Series(data, dtype=float)

    assert any_op(arrow, skipna) == pandas.any(skipna=skipna)

    # Split in the middle and check whether this still works
    if len(data) > 2:
        arrow = pa.chunked_array(
            [data[: len(data) // 2], data[len(data) // 2 :]], type=pa.bool_()
        )
        assert any_op(arrow, skipna) == pandas.any(skipna=skipna)


@settings(deadline=timedelta(milliseconds=1000))
@given(data=st.lists(st.one_of(st.booleans(), st.none())), skipna=st.booleans())
# Test with numpy.array as input.
# This has the caveat that the missing buffer is None.
@example(np.ones(10).astype(bool), False)
@example(np.ones(10).astype(bool), True)
def test_all_op(data, skipna):
    arrow = pa.array(data, type=pa.bool_())
    # https://github.com/pandas-dev/pandas/issues/27709 / https://github.com/pandas-dev/pandas/issues/12863
    pandas = pd.Series(data, dtype=float)

    assert all_op(arrow, skipna) == pandas.all(skipna=skipna)

    # Split in the middle and check whether this still works
    if len(data) > 2:
        arrow = pa.chunked_array(
            [data[: len(data) // 2], data[len(data) // 2 :]], type=pa.bool_()
        )
        assert all_op(arrow, skipna) == pandas.all(skipna=skipna)


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


def test_combined_in_chunk_offsets():
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
