# These unit tests are based upon the work done in Cyberpandas (see NOTICE):
#   https://github.com/ContinuumIO/cyberpandas/blob/master/cyberpandas/test_ip_pandas.py

from pandas.core.internals import ExtensionBlock

import pandas as pd
import pandas.testing as tm
import pandas_string as pd_str
import pyarrow as pa
import pytest


TEST_ARRAY = pa.array(["Test", "string", None])


# ----------------------------------------------------------------------------
# Block Methods
# ----------------------------------------------------------------------------


def test_concatenate_blocks():
    v1 = pd_str.StringArray(TEST_ARRAY)
    s = pd.Series(v1, index=pd.RangeIndex(3), fastpath=True)
    result = pd.concat([s, s], ignore_index=True)
    expected = pd.Series(pd_str.StringArray(pa.array(
        ["Test", "string", None, "Test", "string", None])))
    tm.assert_series_equal(result, expected)

# ----------------------------------------------------------------------------
# Public Constructors
# ----------------------------------------------------------------------------


def test_series_constructor():
    v = pd_str.StringArray(TEST_ARRAY)
    result = pd.Series(v)
    assert result.dtype == v.dtype
    assert isinstance(result._data.blocks[0], ExtensionBlock)


def test_dataframe_constructor():
    v = pd_str.StringArray(TEST_ARRAY)
    df = pd.DataFrame({"A": v})
    assert isinstance(df.dtypes['A'], pd_str.StringDtype)
    assert df.shape == (3, 1)

    # Test some calls to typical DataFrame functions
    str(df)


def test_dataframe_from_series_no_dict():
    s = pd.Series(pd_str.StringArray(TEST_ARRAY))
    result = pd.DataFrame(s)
    expected = pd.DataFrame({0: s})
    tm.assert_frame_equal(result, expected)

    s = pd.Series(pd_str.StringArray(TEST_ARRAY), name='A')
    result = pd.DataFrame(s)
    expected = pd.DataFrame({'A': s})
    tm.assert_frame_equal(result, expected)


def test_dataframe_from_series():
    s = pd.Series(pd_str.StringArray(TEST_ARRAY))
    c = pd.Series(pd.Categorical(['a', 'b']))
    result = pd.DataFrame({"A": s, 'B': c})
    assert isinstance(result.dtypes['A'], pd_str.StringDtype)


def test_getitem_scalar():
    ser = pd.Series(pd_str.StringArray(TEST_ARRAY))
    result = ser[1]
    assert result == "string"


def test_getitem_slice():
    ser = pd.Series(pd_str.StringArray(TEST_ARRAY))
    result = ser[1:]
    expected = pd.Series(pd_str.StringArray(TEST_ARRAY[1:]), index=range(1, 3))
    tm.assert_series_equal(result, expected)


@pytest.mark.xfail(reason="Arrow arrays are not writable")
def test_setitem_scalar():
    ser = pd.Series(pd_str.StringArray(TEST_ARRAY))
    ser[1] = "other_string"
    expected = pd.Series(pd_str.StringArray(pa.array(["Test", "other_string", None])))
    tm.assert_series_equal(ser, expected)


def test_isnull():
    # TODO: test with index
    df = pd.DataFrame({
        "A": pd_str.StringArray(TEST_ARRAY)
    })

    tm.assert_series_equal(df['A'].isnull(), pd.Series([False, False, True], name='A'))


@pytest.mark.xfail
def test_set_index():
    pd.DataFrame({
        'index': [3, 2, 1],
        "A": pd_str.StringArray(TEST_ARRAY)
    }).set_index('index')


def test_copy():
    df = pd.DataFrame({
        "A": pd_str.StringArray(TEST_ARRAY)
    })
    df['A'].copy()
