import itertools
import math
import string
from typing import Optional

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pandas.testing as tm
import pyarrow as pa
import pytest
from hypothesis import given, settings

import fletcher as fr

_basic_string_patterns = [
    ([], ""),
    (["a", "b"], ""),
    (["aa", "ab", "ba"], "a"),
    (["aa", "ab", "ba", "bb", None], "a"),
    (["aa", "ab", "ba", "bb", None], "A"),
    (["aa", "ab", "bA", "bB", None], "a"),
    (["aa", "AB", "ba", "BB", None], "A"),
]


def _gen_string_pattern(
    seq_len=20,
    max_str_len=20,
    max_pat_in_str=3,
    pat_len=3,
    pat=None,
    missing_cnt=0,
    charset="ab",
):
    charset = list(charset)
    if pat is None:
        pat = "".join(np.random.choice(charset, pat_len))
    else:
        pat_len = len(pat)

    if max_str_len < pat_len:
        raise ValueError
    if missing_cnt > seq_len:
        raise ValueError

    min_str_len = math.ceil(max_str_len * 0.5)

    seq = []
    for _ in range(seq_len):
        str_len = np.random.randint(min_str_len, max_str_len + 1)
        base_str = np.random.choice(charset, str_len)

        max_start_i = str_len - pat_len

        if max_start_i > 0:
            pat_cnt_in_cur_str = np.random.randint(max_pat_in_str)
            pat_ind = np.random.randint(0, max_start_i + 1, pat_cnt_in_cur_str)

            for i in pat_ind:
                assert i + pat_len <= str_len
                base_str[i : i + pat_len] = pat

        seq.append("".join(base_str))

    missing_indices = set(np.random.randint(0, seq_len, missing_cnt))
    seq = [None if i in missing_indices else x for i, x in enumerate(seq)]

    return (seq, pat)


def gen_parameter_product(params_dict):
    params_list = list(params_dict.items())

    iter_tuples = itertools.product(*[v for _, v in params_list])
    param_names = [k for k, _ in params_list]

    for test_tuple in iter_tuples:
        yield {name: val for name, val in zip(param_names, test_tuple)}


def _gen_many_string_patterns(seed=1337):
    np.random.seed(seed)

    parameter_combinations = (
        {"seq_len": [1, 2, 20, 30], "pat_len": [1, 5]},
        {"max_str_len": [1, 10, 20, 50]},
        {"max_pat_in_str": [1, 2, 5]},
        {"pat_len": [1, 4, 10]},
        {"pat": ["aab", "aaab", "a", "c"]},
        {"missing_cnt": [0, 4]},
        {"charset": [string.ascii_lowercase]},
    )
    test_set = []
    for params in parameter_combinations:
        for kw in gen_parameter_product(params):
            try:
                test_set.append(_gen_string_pattern(**kw))
            except ValueError:
                # Ignore invalid parameter combinations
                pass

    return test_set


string_patterns = pytest.mark.parametrize("data, pat", _basic_string_patterns)

many_string_patterns = pytest.mark.parametrize(
    "data, pat", _basic_string_patterns + _gen_many_string_patterns()
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


def _check_str_to_t(
    t, func, data, fletcher_variant, test_offset = 0, *args, **kwargs
):
    """Check a .str. function that returns a series with type t."""
    if test_offset > len(data):
        return
    tail_len = len(data) - test_offset

    ser_pd = pd.Series(data, dtype=str).tail(tail_len)
    result_pd = getattr(ser_pd.str, func)(*args, **kwargs)

    ser_fr = _fr_series_from_data(data, fletcher_variant).tail(tail_len)
    result_fr = getattr(ser_fr.fr_text, func)(*args, **kwargs)

    if result_fr.values.data.null_count > 0:
        result_fr = result_fr.astype(object)
    else:
        result_fr = result_fr.astype(t)

    tm.assert_series_equal(result_fr, result_pd)


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


@many_string_patterns
@pytest.mark.parametrize("test_offset", [0, 9])
def test_contains_no_regex_case_sensitive(
    data, pat, test_offset, fletcher_variant
):
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


@many_string_patterns
@pytest.mark.parametrize(
    "repl, n, test_offset", [("len4", -1, 0), ("z", 2, 3), ("", -1, 0)]
)
def test_replace_no_regex_case_sensitive(
    data, pat, repl, n, test_offset, fletcher_variant
):
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


@many_string_patterns
@pytest.mark.parametrize("test_offset", [0, 9])
def test_count_no_regex(
    data, pat, test_offset, fletcher_variant
):
    """Check a .str. function that returns a series with type t."""
    if test_offset > len(data):
        return
    tail_len = len(data) - test_offset

    ser_pd = pd.Series(data, dtype=str).tail(tail_len)
    result_pd = getattr(ser_pd.str, "count")(pat=pat)

    ser_fr = _fr_series_from_data(data, fletcher_variant).tail(tail_len)
    result_fr = getattr(ser_fr.fr_text, "count")(
        pat=pat, case=True, regex=False
    )

    if result_fr.values.data.null_count > 0:
        result_fr = result_fr.astype(np.float64)
    else:
        result_fr = result_fr.astype(np.int64)

    tm.assert_series_equal(result_fr, result_pd)


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
