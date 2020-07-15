from datetime import timedelta
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


@settings(deadline=timedelta(milliseconds=1000))
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
    result_fr = ser_fr.fr_text.cat(ser_fr_other)
    result_fr = result_fr.astype(object)
    # Pandas returns np.nan for NA values in cat, keep this in line
    result_fr[result_fr.isna()] = np.nan
    tm.assert_series_equal(result_fr, result_pd)


def _check_str_to_bool(func, data, fletcher_variant, *args, **kwargs):
    """Check a .str. function that returns a boolean series."""
    ser_pd = pd.Series(data, dtype=str)
    ser_fr = _fr_series_from_data(data, fletcher_variant)

    result_pd = getattr(ser_pd.str, func)(*args, **kwargs)
    result_fr = getattr(ser_fr.fr_text, func)(*args, **kwargs)
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
        result = ser.fr_text.contains(pat, regex=False)
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
    result_fr = ser_fr.fr_text.zfill(max_str_len + 1)
    result_fr = result_fr.astype(object)
    # Pandas returns np.nan for NA values in cat, keep this in line
    result_fr[result_fr.isna()] = np.nan
    tm.assert_series_equal(result_fr, result_pd)


@pytest.mark.parametrize(
    "data, width, expected",
    [
        ([], -1, []),
        ([None], 7, [None]),
        ([""], 3, ["000"]),
        (
            ["a", "_xyz", "aBcDe", "ä", None, "æ", "-01", "+1"],
            4,
            ["000a", "_xyz", "aBcDe", "000ä", None, "000æ", "0-01", "00+1"],
        ),
    ],
)
def test_zfill_simple(data, width, expected, fletcher_variant):
    fr_series = _fr_series_from_data(data, fletcher_variant)
    fr_expected = _fr_series_from_data(expected, fletcher_variant, pa.string())

    for i in range(len(data)):
        result = fr_series.fr_text.zfill(width)
        tm.assert_series_equal(result, fr_expected)
