from datetime import timedelta

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pyarrow as pa
from hypothesis import example, given, settings

from fletcher._algorithms import all_op, any_op


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
    # https://github.com/pandas-dev/pandas/issues/27709 / https://github.com/pandas-dev/pandas/issues/12863
    pandas = pd.Series(data).astype(float)

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
    pandas = pd.Series(data).astype(float)

    assert all_op(arrow, skipna) == pandas.all(skipna=skipna)

    # Split in the middle and check whether this still works
    if len(data) > 2:
        arrow = pa.chunked_array(
            [data[: len(data) // 2], data[len(data) // 2 :]], type=pa.bool_()
        )
        assert all_op(arrow, skipna) == pandas.all(skipna=skipna)
