import math
import string
from typing import Optional, Sequence, Tuple

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pandas.testing as tm
import pyarrow as pa
import pytest
from hypothesis import example, given, settings

import fletcher as fr
from fletcher.testing import examples


@st.composite
def string_patterns_st(draw, max_len=50) -> Tuple[Sequence[Optional[str]], str, int]:
    ab_charset_st = st.sampled_from("ab")
    ascii_charset_st = st.sampled_from(string.ascii_letters)
    charset_st = st.sampled_from((ab_charset_st, ascii_charset_st))
    charset = draw(charset_st)

    fixed_pattern_st = st.sampled_from(["a", "aab", "aabaa"])
    generated_pattern_st = st.text(alphabet=charset, max_size=max_len)
    pattern_st = st.one_of(fixed_pattern_st, generated_pattern_st)
    pattern = draw(pattern_st)

    min_str_size = 0 if len(pattern) > 0 else 1

    raw_str_st = st.one_of(
        st.none(), st.lists(charset, min_size=min_str_size, max_size=max_len)
    )
    raw_seq_st = st.lists(raw_str_st, max_size=max_len)
    raw_seq = draw(raw_seq_st)

    for s in raw_seq:
        if s is None:
            continue

        """
        There seems to be a bug in pandas for this edge case
        >>> pd.Series(['']).str.replace('', 'abc', n=1)
        0
        dtype: object

        But
        >>> pd.Series(['']).str.replace('', 'abc')
        0    abc
        dtype: object

        I believe the second result is the correct one and this is what the
        fletcher implementation returns.
        """

        max_ind = len(s) - len(pattern)
        if max_ind < 0:
            continue
        repl_ind_st = st.integers(min_value=0, max_value=max_ind)
        repl_ind_list_st = st.lists(repl_ind_st, max_size=math.ceil(max_len / 10))

        repl_ind_list = draw(repl_ind_list_st)
        for j in repl_ind_list:
            s[j : j + len(pattern)] = pattern

    seq = ["".join(s) if s is not None else None for s in raw_seq]
    offset = draw(st.integers(min_value=0, max_value=len(seq)))

    return (seq, pattern, offset)


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


def _check_series_equal(result_fr, result_pd):
    result_fr = result_fr.astype(result_pd.dtype)
    tm.assert_series_equal(result_fr, result_pd)


def _check_str_to_t(t, func, data, fletcher_variant, test_offset=0, *args, **kwargs):
    """Check a .str. function that returns a series with type t."""
    tail_len = len(data) - test_offset

    ser_pd = pd.Series(data, dtype=str).tail(tail_len)
    result_pd = getattr(ser_pd.str, func)(*args, **kwargs)

    ser_fr = _fr_series_from_data(data, fletcher_variant).tail(tail_len)
    result_fr = getattr(ser_fr.fr_strx, func)(*args, **kwargs)

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
        result = ser.fr_strx.contains(pat, regex=False)
        tm.assert_series_equal(result, expected)


@settings(deadline=None)
@given(data_tuple=string_patterns_st())
def test_contains_no_regex_case_sensitive(data_tuple, fletcher_variant):
    data, pat, test_offset = data_tuple
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


@settings(deadline=None)
@given(
    data_tuple=string_patterns_st(),
    n=st.integers(min_value=0, max_value=10),
    repl=st.sampled_from(["len4", "", "z"]),
)
@example(
    data_tuple=(["aababaa"], "aabaa", 0),
    repl="len4",
    n=1,
    fletcher_variant="continuous",
)
@example(data_tuple=(["aaa"], "a", 0), repl="len4", n=1, fletcher_variant="continuous")
def test_replace_no_regex_case_sensitive(data_tuple, repl, n, fletcher_variant):
    data, pat, test_offset = data_tuple
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
@given(data_tuple=string_patterns_st())
@example(data_tuple=(["a"], "", 0), fletcher_variant="chunked")
def test_count_no_regex(data_tuple, fletcher_variant):
    """Check a .str. function that returns a series with type t."""
    data, pat, test_offset = data_tuple

    tail_len = len(data) - test_offset

    ser_pd = pd.Series(data, dtype=str).tail(tail_len)
    result_pd = getattr(ser_pd.str, "count")(pat=pat)

    ser_fr = _fr_series_from_data(data, fletcher_variant).tail(tail_len)
    result_fr = getattr(ser_fr.fr_strx, "count")(pat=pat, case=True, regex=False)

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
    result_fr = ser_fr.fr_strx.zfill(max_str_len + 1)
    result_fr = result_fr.astype(object)
    # Pandas returns np.nan for NA values in cat, keep this in line
    result_fr[result_fr.isna()] = np.nan
    tm.assert_series_equal(result_fr, result_pd)


@settings(deadline=None, max_examples=3)
@given(data=st.lists(st.one_of(st.text(), st.none())))
@examples(
    example_list=[
        [
            " 000000000000000000000000000000000000000000İࠀࠀࠀࠀ𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐤱000000000000𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀𐀀"
        ],
        ["\x80 "],
        [],
    ],
    example_kword="data",
)
def test_text_strip_offset(fletcher_variant, fletcher_slice_offset, data):
    _do_test_text_strip(fletcher_variant, fletcher_slice_offset, data)


@settings(deadline=None)
@given(data=st.lists(st.one_of(st.text(), st.none())))
@examples(
    example_list=[
        [],
        [""],
        [None],
        [" "],
        ["\u2000"],
        [" a"],
        ["a "],
        [" a "],
        # https://github.com/xhochy/fletcher/issues/174
        ["\xa0"],
        ["\u2000a\u2000"],
        ["\u2000\u200C\u2000"],
        ["\n\u200C\r"],
        ["\u2000\x80\u2000"],
        ["\t\x80\x0b"],
        ["\u2000\u10FFFF\u2000"],
        [" \u10FFFF "],
    ]
    + [
        [c]
        for c in " \t\r\n\x1f\x1e\x1d\x1c\x0c\x0b"
        "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2000\u2009\u200A\u200B\u2028\u2029\u202F\u205F"
    ]
    + [[chr(c)] for c in range(0x32)]
    + [[chr(c)] for c in range(0x80, 0x85)]
    + [[chr(c)] for c in range(0x200C, 0x2030)]
    + [[chr(c)] for c in range(0x2060, 0x2070)]
    + [[chr(c)] for c in range(0x10FFFE, 0x110000)],
    example_kword="data",
)
def test_text_strip(fletcher_variant, data):
    _do_test_text_strip(fletcher_variant, 1, data)


def _do_test_text_strip(fletcher_variant, fletcher_slice_offset, data):
    if any("\x00" in x for x in data if x):
        # pytest.skip("pandas cannot handle \\x00 characters in tests")
        # Skip is not working properly with hypothesis
        return
    ser_pd = pd.Series(data, dtype=str)
    arrow_data = pa.array(
        [None for _ in range(fletcher_slice_offset)] + data, type=pa.string()
    )
    if fletcher_variant == "chunked":
        fr_array = fr.FletcherChunkedArray(arrow_data)
    else:
        fr_array = fr.FletcherContinuousArray(arrow_data)
    ser_fr = pd.Series(fr_array[fletcher_slice_offset:])

    result_pd = ser_pd.str.strip()
    result_fr = ser_fr.fr_strx.strip()
    result_fr = result_fr.astype(object)
    # Pandas returns np.nan for NA values in cat, keep this in line
    result_fr[result_fr.isna()] = np.nan
    result_pd[result_pd.isna()] = np.nan
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


@settings(deadline=None)
@given(char=st.characters(blacklist_categories=("Cs",)))
def test_utf8_size(char):
    char_bytes = char.encode("utf-8")
    expected = len(char_bytes)
    computed = fr.algorithms.string.get_utf8_size(char_bytes[0])

    assert computed == expected
