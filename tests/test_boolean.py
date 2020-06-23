from datetime import timedelta

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pyarrow as pa
from hypothesis import example, given, settings

from fletcher.algorithms.bool import all_op, any_op


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


def test_np_any(fletcher_array):
    arr = fletcher_array([True, False, None])
    assert np.any(arr)

    arr = fletcher_array([True, False, True])
    assert np.any(arr)

    # TODO(pandas-0.26): Uncomment this when BooleanArray landed.
    #   Then we change the behaviour.
    # arr = fr.FletcherChunkedArray([False, False, None])
    # assert np.any(arr) is pd.NA

    arr = fletcher_array([False, False, False])
    assert not np.any(arr)


def test_or(fletcher_array):
    # Scalar versions
    # non-null versions
    result = fletcher_array([True, False]) | pd.NA
    expected = fletcher_array([True, None])
    pdt.assert_extension_array_equal(result, expected)

    result = fletcher_array([True, False, None]) | pd.NA
    expected = fletcher_array([True, None, None])
    pdt.assert_extension_array_equal(result, expected)

    result = fletcher_array([True, False, None]) | True
    expected = fletcher_array([True, True, True])
    pdt.assert_extension_array_equal(result, expected)

    result = fletcher_array([True, False, None]) | False
    expected = fletcher_array([True, False, None])
    pdt.assert_extension_array_equal(result, expected)

    # Array version
    # Non-null version
    result = fletcher_array([True, False, False]) | fletcher_array([False, True, False])
    expected = fletcher_array([True, True, False])
    pdt.assert_extension_array_equal(result, expected)
    # One has nulls, the other not
    result = fletcher_array([True, False, None, None]) | fletcher_array(
        [False, True, False, True]
    )
    expected = fletcher_array([True, True, None, True])
    pdt.assert_extension_array_equal(result, expected)
    # Both have nulls
    result = fletcher_array([True, False, None, None]) | fletcher_array(
        [None, True, False, True]
    )
    pdt.assert_extension_array_equal(result, expected)

    result = fletcher_array([True, False, None, None]) | np.array(
        [False, True, False, True]
    )
    pdt.assert_extension_array_equal(result, expected)


def test_ior(fletcher_array):
    # Needed to support .replace()
    # ior is |=

    # Scalar version
    arr = fletcher_array([True, False, None])
    arr |= True
    expected = fletcher_array([True, True, True])
    pdt.assert_extension_array_equal(arr, expected)

    # Array version
    arr = fletcher_array([True, False, None, None])
    arr |= fletcher_array([False, True, False, True])
    expected = fletcher_array([True, True, None, True])
    pdt.assert_extension_array_equal(arr, expected)
