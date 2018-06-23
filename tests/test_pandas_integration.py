# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

# These unit tests are based upon the work done in Cyberpandas (see NOTICE):
#   https://github.com/ContinuumIO/cyberpandas/blob/master/cyberpandas/test_ip_pandas.py

from pandas.core.internals import ExtensionBlock

import fletcher as fr
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as tm
import pyarrow as pa
import pytest

TEST_LIST = ["Test", "string", None]
TEST_ARRAY = pa.array(TEST_LIST)


# ----------------------------------------------------------------------------
# Block Methods
# ----------------------------------------------------------------------------


def test_concatenate_blocks():
    v1 = fr.StringArray(TEST_ARRAY)
    s = pd.Series(v1, index=pd.RangeIndex(3), fastpath=True)
    result = pd.concat([s, s], ignore_index=True)
    expected = pd.Series(
        fr.StringArray(pa.array(["Test", "string", None, "Test", "string", None]))
    )
    tm.assert_series_equal(result, expected)


# ----------------------------------------------------------------------------
# Public Constructors
# ----------------------------------------------------------------------------


def test_series_constructor():
    v = fr.StringArray(TEST_ARRAY)
    result = pd.Series(v)
    assert result.dtype == v.dtype
    assert isinstance(result._data.blocks[0], ExtensionBlock)


def test_dataframe_constructor():
    v = fr.StringArray(TEST_ARRAY)
    df = pd.DataFrame({"A": v})
    assert isinstance(df.dtypes["A"], fr.FletcherDtype)
    assert df.shape == (3, 1)

    # Test some calls to typical DataFrame functions
    str(df)
    df.info()


def test_dataframe_from_series_no_dict():
    s = pd.Series(fr.StringArray(TEST_ARRAY))
    result = pd.DataFrame(s)
    expected = pd.DataFrame({0: s})
    tm.assert_frame_equal(result, expected)

    s = pd.Series(fr.StringArray(TEST_ARRAY), name="A")
    result = pd.DataFrame(s)
    expected = pd.DataFrame({"A": s})
    tm.assert_frame_equal(result, expected)


def test_dataframe_from_series():
    s = pd.Series(fr.StringArray(TEST_ARRAY))
    c = pd.Series(pd.Categorical(["a", "b"]))
    result = pd.DataFrame({"A": s, "B": c})
    assert isinstance(result.dtypes["A"], fr.FletcherDtype)


def test_getitem_scalar():
    ser = pd.Series(fr.StringArray(TEST_ARRAY))
    result = ser[1]
    assert result == "string"


def test_getitem_slice():
    ser = pd.Series(fr.StringArray(TEST_ARRAY))
    result = ser[1:]
    expected = pd.Series(fr.StringArray(TEST_ARRAY[1:]), index=range(1, 3))
    tm.assert_series_equal(result, expected)


@pytest.mark.xfail(reason="Arrow arrays are not writable")
def test_setitem_scalar():
    ser = pd.Series(fr.StringArray(TEST_ARRAY))
    ser[1] = "other_string"
    expected = pd.Series(fr.StringArray(pa.array(["Test", "other_string", None])))
    tm.assert_series_equal(ser, expected)


def test_isnull():
    df = pd.DataFrame({"A": fr.StringArray(TEST_ARRAY)})

    tm.assert_series_equal(df["A"].isnull(), pd.Series([False, False, True], name="A"))


def test_set_index():
    pd.DataFrame({"index": [3, 2, 1], "A": fr.StringArray(TEST_ARRAY)}).set_index(
        "index"
    )


def test_copy():
    df = pd.DataFrame({"A": fr.StringArray(TEST_ARRAY)})
    df["A"].copy()


def test_nbytes():
    array = fr.StringArray(pa.array(["A", None, "CC"]))
    # Minimal storage usage:
    # 1 byte for the valid bitmap
    # 4 bytes for the offset array
    # 3 bytes for the actual string content
    assert array.nbytes >= 8


def test_series_attributes():
    s = pd.Series(fr.StringArray(TEST_ARRAY))
    assert s.ndim == 1
    assert s.size == 3
    assert s.base is not None
    assert (s.T == s).all()
    assert s.memory_usage() > 8


def test_isna():
    s = pd.Series(fr.StringArray(TEST_ARRAY))
    expected = pd.Series([False, False, True])
    tm.assert_series_equal(s.isna(), expected)
    tm.assert_series_equal(s.notna(), ~expected)


def test_np_asarray():
    s = pd.Series(fr.StringArray(TEST_ARRAY))
    expected = np.asarray(TEST_LIST)
    npt.assert_array_equal(np.asarray(s), expected)


def test_astype_object():
    s = pd.Series(fr.StringArray(TEST_ARRAY))
    expected = pd.Series(TEST_LIST)
    tm.assert_series_equal(s.astype(object), expected)


def test_factorize():
    arr = fr.StringArray(TEST_ARRAY)
    labels, uniques = arr.factorize()
    expected_labels, expected_uniques = pd.factorize(arr.astype(object))

    assert isinstance(uniques, fr.StringArray)

    uniques = uniques.astype(object)
    npt.assert_array_equal(labels, expected_labels)
    npt.assert_array_equal(uniques, expected_uniques)


def test_groupby():
    arr = fr.StringArray(["a", "a", "b", None])
    df = pd.DataFrame({"str": arr, "int": [10, 5, 24, 6]})
    result = df.groupby("str").sum()

    expected = pd.DataFrame({"int": [15, 24]}, index=pd.Index(["a", "b"], name="str"))
    tm.assert_frame_equal(result, expected)


@pytest.mark.xfail
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("kind", ["quicksort", "mergesort", "heapsort"])
def test_argsort(ascending, kind):
    s = pd.Series(fr.StringArray(TEST_ARRAY))
    result = s.argsort(ascending=ascending, kind=kind)
    expected = s.astype(object).argsort(ascending=ascending, kind=kind)
    tm.assert_frame_equal(result, expected)
