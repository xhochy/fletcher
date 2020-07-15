import math
import string
from typing import Optional

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pandas.testing as tm
import pyarrow as pa
import pytest
from hypothesis import assume, example, given, settings

import fletcher as fr


@st.composite
def string_patterns_st(draw, max_len=50):
    ab_charset_st = st.sampled_from("ab")
    ascii_charset_st = st.sampled_from(string.ascii_letters)
    charset_st = st.sampled_from((ab_charset_st, ascii_charset_st))
    charset = draw(charset_st)

    fixed_pattern_st = st.sampled_from(["a", "aab", "aabaa"])
    generated_pattern_st = st.text(alphabet=charset, max_size=max_len)
    pattern_st = st.one_of(fixed_pattern_st, generated_pattern_st)
    pattern = draw(pattern_st)

    raw_str_st = st.one_of(st.none(), st.lists(charset, max_size=max_len))
    raw_seq_st = st.lists(raw_str_st, max_size=max_len)
    raw_seq = draw(raw_seq_st)

    assume(any(s is not None for s in raw_seq))

    for s in raw_seq:
        if s is None:
            continue

        """
        There seems to be a bug in pandas for this edge case
        >>> pd.Series(['']).str.replace('', 'abc', n=1))
        0
        dtype: object

        But
        >>> pd.Series(['']).str.replace('', 'abc'))
        0    abc
        dtype: object

        I believe the second result is the correct one and this is what the
        fletcher implementation returns.
        """

        assume(len(s) > 0 or len(pattern) > 0)

        max_ind = len(s) - len(pattern)
        if max_ind < 0:
            continue
        repl_ind_st = st.integers(min_value=0, max_value=max_ind)
        repl_ind_list_st = st.lists(repl_ind_st, max_size=math.ceil(max_len / 10))

        repl_ind_list = draw(repl_ind_list_st)
        for j in repl_ind_list:
            s[j : j + len(pattern)] = pattern

    seq = ["".join(s) if s is not None else None for s in raw_seq]

    return (seq, pattern)


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
    result_fr = ser_fr.fr_text.cat(ser_fr_other)
    result_fr = result_fr.astype(object)
    # Pandas returns np.nan for NA values in cat, keep this in line
    result_fr[result_fr.isna()] = np.nan
    tm.assert_series_equal(result_fr, result_pd)


def _check_series_equal(result_fr, result_pd):
    result_fr = result_fr.astype(result_pd.dtype)
    tm.assert_series_equal(result_fr, result_pd)


def _check_str_to_t(t, func, data, fletcher_variant, test_offset=0, *args, **kwargs):
    """Check a .str. function that returns a series with type t."""
    tail_len = len(data) - test_offset

    ser_pd = pd.Series(data, dtype=str).tail(tail_len)
    result_pd = getattr(ser_pd.str, func)(*args, **kwargs)

    ser_fr = _fr_series_from_data(data, fletcher_variant).tail(tail_len)
    result_fr = getattr(ser_fr.fr_text, func)(*args, **kwargs)

    _check_series_equal(result_fr, result_pd)


def _check_str_to_str(func, data, fletcher_variant, *args, **kwargs):
    _check_str_to_t(str, func, data, fletcher_variant, *args, **kwargs)


def _check_str_to_bool(func, data, fletcher_variant, *args, **kwargs):
    _check_str_to_t(bool, func, data, fletcher_variant, *args, **kwargs)


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


# @settings(deadline=None)
@given(
    data_pat_tuple=string_patterns_st(),
    test_offset=st.integers(min_value=0, max_value=15),
)
def test_contains_no_regex_case_sensitive(
    data_pat_tuple, test_offset, fletcher_variant
):
    data, pat = data_pat_tuple
    assume(test_offset < len(data))
    _check_str_to_bool(
        "contains",
        data,
        fletcher_variant,
        test_offset=test_offset,
        pat=pat,
        case=True,
        regex=False,
    )


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


# @many_string_patterns
# @pytest.mark.parametrize(
#    "repl, n, test_offset", [("len4", -1, 0), ("z", 2, 3), ("", -1, 0)]
# )


@settings(deadline=None)
@given(
    data_pat_tuple=string_patterns_st(),
    test_offset=st.integers(min_value=0, max_value=15),
    n=st.integers(min_value=0, max_value=10),
    repl=st.sampled_from(["len4", "", "z"]),
)
@example(
    data_pat_tuple=(["aababaa"], "aabaa"),
    repl="len4",
    n=1,
    test_offset=0,
    fletcher_variant="chunked",
)
def test_replace_no_regex_case_sensitive(
    data_pat_tuple, repl, n, test_offset, fletcher_variant
):
    data, pat = data_pat_tuple
    assume(len(data) > test_offset)
    _check_str_to_str(
        "replace",
        data,
        fletcher_variant,
        test_offset=test_offset,
        pat=pat,
        repl=repl,
        n=n,
        case=True,
        regex=False,
    )


@settings(deadline=None)
@given(
    data_pat_tuple=string_patterns_st(),
    test_offset=st.integers(min_value=0, max_value=15),
)
def test_count_no_regex(data_pat_tuple, test_offset, fletcher_variant):
    """Check a .str. function that returns a series with type t."""
    data, pat = data_pat_tuple

    assume(test_offset < len(data))
    tail_len = len(data) - test_offset

    ser_pd = pd.Series(data, dtype=str).tail(tail_len)
    result_pd = getattr(ser_pd.str, "count")(pat=pat)

    ser_fr = _fr_series_from_data(data, fletcher_variant).tail(tail_len)
    result_fr = getattr(ser_fr.fr_text, "count")(pat=pat, case=True, regex=False)

    _check_series_equal(result_fr, result_pd)


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
