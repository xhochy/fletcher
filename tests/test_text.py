from typing import Optional

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pandas.testing as tm
import pyarrow as pa
import pytest
from hypothesis import given, settings

import fletcher as fr

string_patterns = pytest.mark.parametrize(
    "data, pat",
    [
        ([], ""),
        (["a", "b"], ""),
        (["aa", "ab", "ba"], "a"),
        (["aa", "ab", "ba", "bb", None], "a"),
        (["aa", "ab", "ba", "bb", None], "A"),
        (["aa", "ab", "bA", "bB", None], "a"),
        (["aa", "AB", "ba", "BB", None], "A"),
    ],
)


def _fr_series_from_data(data, fletcher_variant, dtype=pa.string()):
    arrow_data = pa.array(data, type=dtype)
    if fletcher_variant == "chunked":
        fr_array = fr.FletcherChunkedArray(arrow_data)
    else:
        fr_array = fr.FletcherContinuousArray(arrow_data)
    return pd.Series(fr_array)


@settings(deadline=None)
@given(data=st.lists(st.one_of(st.text(), st.none())))
def test_text_cat(data, fletcher_variant, fletcher_variant_2):
    if any("\x00" in x for x in data if x):
        # pytest.skip("pandas cannot handle \\x00 characters in tests")
        # Skip is not working properly with hypothesis
        return
    ser_pd = pd.Series(data, dtype=str)
    ser_fr = _fr_series_from_data(data, fletcher_variant)
    ser_fr_other = _fr_series_from_data(data, fletcher_variant_2)

    result_pd = ser_pd.str.cat(ser_pd)
    result_fr = ser_fr.fr_strx.cat(ser_fr_other)
    result_fr = result_fr.astype(object)
    # Pandas returns np.nan for NA values in cat, keep this in line
    result_fr[result_fr.isna()] = np.nan
    tm.assert_series_equal(result_fr, result_pd)


def _check_str_to_bool(func, data, fletcher_variant, *args, **kwargs):
    """Check a .str. function that returns a boolean series."""
    ser_pd = pd.Series(data, dtype=str)
    ser_fr = _fr_series_from_data(data, fletcher_variant)

    result_pd = getattr(ser_pd.str, func)(*args, **kwargs)
    result_fr = getattr(ser_fr.fr_strx, func)(*args, **kwargs)
    if result_fr.values.data.null_count > 0:
        result_fr = result_fr.astype(object)
    else:
        result_fr = result_fr.astype(bool)
    tm.assert_series_equal(result_fr, result_pd)


@string_patterns
def test_text_endswith(data, pat, fletcher_variant):
    _check_str_to_bool("endswith", data, fletcher_variant, pat=pat)


@string_patterns
def test_text_startswith(data, pat, fletcher_variant):
    _check_str_to_bool("startswith", data, fletcher_variant, pat=pat)


@string_patterns
def test_contains_no_regex(data, pat, fletcher_variant):
    _check_str_to_bool("contains", data, fletcher_variant, pat=pat, regex=False)


@pytest.mark.parametrize(
    "data, pat, expected",
    [
        ([], "", []),
        (["a", "b"], "", [True, True]),
        (["aa", "Ab", "ba", "bb", None], "a", [True, False, True, False, None]),
    ],
)
def test_contains_no_regex_ascii(data, pat, expected, fletcher_variant):
    fr_series = _fr_series_from_data(data, fletcher_variant)
    fr_expected = _fr_series_from_data(expected, fletcher_variant, pa.bool_())

    # Run over slices to check offset handling code
    for i in range(len(data)):
        ser = fr_series.tail(len(data) - i)
        expected = fr_expected.tail(len(data) - i)
        result = ser.fr_strx.contains(pat, regex=False)
        tm.assert_series_equal(result, expected)


@string_patterns
def test_contains_no_regex_ignore_case(data, pat, fletcher_variant):
    _check_str_to_bool(
        "contains", data, fletcher_variant, pat=pat, regex=False, case=False
    )


regex_patterns = pytest.mark.parametrize(
    "data, pat",
    [
        ([], ""),
        (["a", "b"], ""),
        (["aa", "ab", "ba"], "a"),
        (["aa", "ab", "ba", None], "a"),
        (["aa", "ab", "ba", None], "a$"),
        (["aa", "ab", "ba", None], "^a"),
        (["Aa", "ab", "ba", None], "A"),
        (["aa", "AB", "ba", None], "A$"),
        (["aa", "AB", "ba", None], "^A"),
    ],
)


@regex_patterns
def test_contains_regex(data, pat, fletcher_variant):
    _check_str_to_bool("contains", data, fletcher_variant, pat=pat, regex=True)


@regex_patterns
def test_contains_regex_ignore_case(data, pat, fletcher_variant):
    _check_str_to_bool(
        "contains", data, fletcher_variant, pat=pat, regex=True, case=False
    )


def _optional_len(x: Optional[str]) -> int:
    if x is not None:
        return len(x)
    else:
        return 0


@settings(deadline=None)
@given(data=st.lists(st.one_of(st.text(), st.none())))
def test_text_zfill(data, fletcher_variant):
    if any("\x00" in x for x in data if x):
        # pytest.skip("pandas cannot handle \\x00 characters in tests")
        # Skip is not working properly with hypothesis
        return
    ser_pd = pd.Series(data, dtype=str)
    max_str_len = ser_pd.map(_optional_len).max()
    if pd.isna(max_str_len):
        max_str_len = 0
    arrow_data = pa.array(data, type=pa.string())
    if fletcher_variant == "chunked":
        fr_array = fr.FletcherChunkedArray(arrow_data)
    else:
        fr_array = fr.FletcherContinuousArray(arrow_data)
    ser_fr = pd.Series(fr_array)

    result_pd = ser_pd.str.zfill(max_str_len + 1)
    result_fr = ser_fr.fr_strx.zfill(max_str_len + 1)
    result_fr = result_fr.astype(object)
    # Pandas returns np.nan for NA values in cat, keep this in line
    result_fr[result_fr.isna()] = np.nan
    tm.assert_series_equal(result_fr, result_pd)


def test_fr_str_accessor(fletcher_array):
    data = ["a", "b"]
    ser_pd = pd.Series(data)

    # object series is returned
    s = ser_pd.fr_str.encode("utf8")
    assert s.dtype == np.dtype("O")

    # test fletcher functionality and fallback to pandas
    arrow_data = pa.array(data, type=pa.string())
    fr_array = fletcher_array(arrow_data)
    ser_fr = pd.Series(fr_array)
    # pandas strings only method
    s = ser_fr.fr_str.encode("utf8")
    assert isinstance(s.values, fr.FletcherBaseArray)


def test_fr_str_accessor_fail(fletcher_variant):

    data = [1, 2]
    ser_pd = pd.Series(data)

    with pytest.raises(Exception):
        ser_pd.fr_str.startswith("a")


# @pytest.mark.parametrize(
#     ["data", "slice_", "expected"],
#     [
#         (["abcd", "defg", "hijk"], (1, 2, 1), ["b", "e", "i"]),
#         (["abcd", "defg", "h"], (1, 2, 1), ["b", "e", ""]),
#         (["abcd", "defg", "hijk"], (1, 4, 2), ["bd", "eg", "ik"]),
#         (["abcd", "defg", "hijk"], (0, -2, 1), ["ab", "de", "hi"]),
#         (["abcd", "defg", "hijk"], (-5, -2, 1), ["ab", "de", "hi"]),
#         (["aécd", "d🙂fg", "écijk"], (1, 2, 1), ["é", "🙂", "c"]),
#         (["abcd", "defg", "hijk"], (3, 1, -1), ["dc", "gf", "kj"]),
#         (["abcd", "defg", "hijk"], (5, 0, -2), ["db", "ge", "ki"]),
#         (["abcd", "defg", "hijk"], (3, 20, 1), ["d", "g", "k"]),
#         (["abcd", "defg", "hijk"], (10, 20, 1), ["", "", ""]),
#         (["abcd", "defg", None], (10, 20, 1), ["", "", None]),
#         (["abcd", "defg", "hijk"], (1, None, 1), ["bcd", "efg", "ijk"]),
#     ],
# )
# @pytest.mark.parametrize(
#     "storage_type", [fr.FletcherContinuousArray, fr.FletcherChunkedArray]
# )
# def test_slice(data, slice_, expected, storage_type):
#     fr_series = pd.Series(storage_type(data))
#     fr_out = fr_series.fr_str.slice(*slice_).astype(object)
#     pd.testing.assert_series_equal(fr_out, pd.Series(expected))

#     pd_out = pd.Series(data).str.slice(*slice_)
#     pd.testing.assert_series_equal(fr_out, pd_out)


@settings(deadline=None)
@given(
    data=st.lists(st.one_of(st.text(), st.none())),
    slice_=st.tuples(st.integers(-20, 20), st.integers(-20, 20), st.integers(-20, 20)),
)
def test_slice(data, slice_, fletcher_variant):
    if slice_[2] == 0:
        pytest.raises(ValueError)
        return
    if data == [None] or data == [""]:
        return

    ser_fr = _fr_series_from_data(data, fletcher_variant)
    result_fr = ser_fr.fr_str.slice(*slice_)
    result_fr = result_fr.astype(object)
    # Pandas returns np.nan for NA values in cat, keep this in line
    result_fr[result_fr.isna()] = np.nan

    ser_pd = pd.Series(data, dtype=object)
    result_pd = ser_pd.str.slice(*slice_)

    tm.assert_series_equal(result_fr, result_pd)
