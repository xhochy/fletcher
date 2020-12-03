import math
import string
import sys
from typing import Optional, Sequence, Tuple

import hypothesis.strategies as st
import numba
import numpy as np
import pandas as pd
import pandas.testing as tm
import pyarrow as pa
import pytest
from hypothesis import example, given, settings

import fletcher as fr
from fletcher.algorithms.string import apply_binary_str
from fletcher.testing import examples

try:
    # Only available in pandas 1.2+
    # When this class is defined, we can also use `.str` on fletcher columns.
    from pandas.core.strings.object_array import ObjectStringArrayMixin  # noqa F401

    _str_accessors = ["str", "fr_str"]
except ImportError:
    _str_accessors = ["fr_str"]


# Remove Lm once https://github.com/JuliaStrings/utf8proc/pull/196 has been released.
# FIXME: Keep for now at max_codepoint 255 as we would otherwise need a lot of exclusions. Widen once we have a utf8proc release or we have dropped Python 3.6.
st_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Lm"), max_codepoint=255)
)


def supported_by_utf8proc(s):
    """Check a string whether all characters are supported by utf8proc."""
    # See https://github.com/JuliaStrings/utf8proc/pull/196
    return len(set("'\U0002ceb0\U00018af4ᴬᴭªᶛº¹²³").intersection(s)) == 0


def supported_by_python(s):
    """Check a string whether all characters are supported by Python."""
    if sys.version_info < (3, 7):
        # Needs unicode 10
        if any(ord(c) > 110592 for c in s):
            return False
        if any(ord(c) >= 69632 and ord(c) <= 73727 for c in s):
            return False
        if set("🄰ꭜ").intersection(s):
            return False
    if sys.version_info < (3, 8):
        if any(ord(c) > 183983 for c in s):
            # Needs Unicode 12+
            return False
    # Probably needs unicode 13, check with Python 3.9 in unicodedata.unidata_version
    if set("\U00018af5\U00018af4\U00018af3").intersection(s):
        return False
    if any(ord(c) > 196607 for c in s):
        return False
    return True


def filter_supported(strings):
    return [
        s
        for s in strings
        if s is None or (supported_by_python(s) and supported_by_utf8proc(s))
    ]


@pytest.fixture(params=_str_accessors, scope="module")
def str_accessor(request):
    return request.param


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

strip_examples = examples(
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


def _fr_series_from_data(data, fletcher_variant, dtype=pa.string(), index=None):
    arrow_data = pa.array(data, type=dtype)
    if fletcher_variant == "chunked":
        fr_array = fr.FletcherChunkedArray(arrow_data)
    else:
        fr_array = fr.FletcherContinuousArray(arrow_data)
    return pd.Series(fr_array, index=index)


def _check_series_equal(result_fr, result_pd):
    result_fr = result_fr.astype(result_pd.dtype)
    tm.assert_series_equal(result_fr, result_pd)


def _check_str_to_t(
    t, func, data, str_accessor, fletcher_variant, test_offset=0, *args, **kwargs
):
    """Check a .str. function that returns a series with type t."""
    tail_len = len(data) - test_offset

    error = None
    try:
        ser_pd = pd.Series(data, dtype=str).tail(tail_len)
        result_pd = getattr(ser_pd.str, func)(*args, **kwargs)
    except Exception as e:
        error = e

    ser_fr = _fr_series_from_data(data, fletcher_variant).tail(tail_len)
    if error:
        # If pandas raises an exception, fletcher should do so, too.
        with pytest.raises(type(error)):
            result_fr = getattr(getattr(ser_fr, str_accessor), func)(*args, **kwargs)
    else:
        result_fr = getattr(getattr(ser_fr, str_accessor), func)(*args, **kwargs)
        _check_series_equal(result_fr, result_pd)


def _check_str_to_str(func, data, str_accessor, fletcher_variant, *args, **kwargs):
    _check_str_to_t(str, func, data, str_accessor, fletcher_variant, *args, **kwargs)


def _check_str_to_bool(func, data, str_accessor, fletcher_variant, *args, **kwargs):
    _check_str_to_t(bool, func, data, str_accessor, fletcher_variant, *args, **kwargs)


def _check_str_to_int(func, data, str_accessor, fletcher_variant, *args, **kwargs):
    _check_str_to_t(int, func, data, str_accessor, fletcher_variant, *args, **kwargs)


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
@given(char=st.characters(blacklist_categories=("Cs",)))
def test_utf8_size(char):
    char_bytes = char.encode("utf-8")
    expected = len(char_bytes)
    computed = fr.algorithms.string.get_utf8_size(char_bytes[0])

    assert computed == expected


#####################################################
## String accessor methods (sorted alphabetically) ##
#####################################################


@settings(deadline=None)
@given(data=st.lists(st.one_of(st_text, st.none())))
def test_capitalize(data, str_accessor, fletcher_variant):
    _check_str_to_str("capitalize", data, str_accessor, fletcher_variant)


@settings(deadline=None)
@given(data=st.lists(st.one_of(st_text, st.none())))
def test_casefold(data, str_accessor, fletcher_variant):
    _check_str_to_str("casefold", data, str_accessor, fletcher_variant)


@settings(deadline=None)
@given(data=st.lists(st.one_of(st_text, st.none())))
def test_cat(data, str_accessor, fletcher_variant, fletcher_variant_2):
    if any("\x00" in x for x in data if x):
        # pytest.skip("pandas cannot handle \\x00 characters in tests")
        # Skip is not working properly with hypothesis
        return
    ser_pd = pd.Series(data, dtype=str)
    ser_fr = _fr_series_from_data(data, fletcher_variant)
    ser_fr_other = _fr_series_from_data(data, fletcher_variant_2)

    result_pd = ser_pd.str.cat(ser_pd)
    result_fr = getattr(ser_fr, str_accessor).cat(ser_fr_other)
    result_fr = result_fr.astype(object)
    # Pandas returns np.nan for NA values in cat, keep this in line
    result_fr[result_fr.isna()] = np.nan
    tm.assert_series_equal(result_fr, result_pd)


@settings(deadline=None)
@given(
    data=st.lists(st.one_of(st_text, st.none())),
    width=st.integers(min_value=0, max_value=50),
)
def test_center(data, width, str_accessor, fletcher_variant):
    _check_str_to_str("center", data, str_accessor, fletcher_variant, width=width)


@string_patterns
def test_contains_no_regex(data, pat, str_accessor, fletcher_variant):
    _check_str_to_bool(
        "contains", data, str_accessor, fletcher_variant, pat=pat, regex=False
    )


@pytest.mark.parametrize(
    "data, pat, expected",
    [
        ([], "", []),
        (["a", "b"], "", [True, True]),
        (["aa", "Ab", "ba", "bb", None], "a", [True, False, True, False, None]),
    ],
)
def test_contains_no_regex_ascii(data, pat, expected, str_accessor, fletcher_variant):
    if str_accessor == "str":
        pytest.skip(
            "return types not stable yet, might sometimes return null instead of bool"
        )
        return
    fr_series = _fr_series_from_data(data, fletcher_variant)
    fr_expected = _fr_series_from_data(expected, fletcher_variant, pa.bool_())

    # Run over slices to check offset handling code
    for i in range(len(data)):
        ser = fr_series.tail(len(data) - i)
        expected = fr_expected.tail(len(data) - i)
        result = getattr(ser, str_accessor).contains(pat, regex=False)
        tm.assert_series_equal(result, expected)


@settings(deadline=None)
@given(data_tuple=string_patterns_st())
def test_contains_no_regex_case_sensitive(data_tuple, str_accessor, fletcher_variant):
    data, pat, test_offset = data_tuple
    _check_str_to_bool(
        "contains",
        data,
        str_accessor,
        fletcher_variant,
        test_offset=test_offset,
        pat=pat,
        case=True,
        regex=False,
    )


@string_patterns
def test_contains_no_regex_ignore_case(data, pat, str_accessor, fletcher_variant):
    _check_str_to_bool(
        "contains",
        data,
        str_accessor,
        fletcher_variant,
        pat=pat,
        regex=False,
        case=False,
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
def test_contains_regex(data, pat, str_accessor, fletcher_variant):
    _check_str_to_bool(
        "contains", data, str_accessor, fletcher_variant, pat=pat, regex=True
    )


@regex_patterns
def test_contains_regex_ignore_case(data, pat, str_accessor, fletcher_variant):
    _check_str_to_bool(
        "contains",
        data,
        str_accessor,
        fletcher_variant,
        pat=pat,
        regex=True,
        case=False,
    )


@settings(deadline=None)
@given(data_tuple=string_patterns_st())
@example(data_tuple=(["a"], "", 0), fletcher_variant="chunked")
def test_count_no_regex(data_tuple, str_accessor, fletcher_variant):
    """Check a .str. function that returns a series with type t."""
    data, pat, test_offset = data_tuple

    tail_len = len(data) - test_offset

    ser_pd = pd.Series(data, dtype=str).tail(tail_len)
    result_pd = getattr(ser_pd.str, "count")(pat=pat)

    ser_fr = _fr_series_from_data(data, fletcher_variant).tail(tail_len)
    kwargs = {}
    if str_accessor.startswith("fr_"):
        kwargs["regex"] = False
    result_fr = getattr(ser_fr, str_accessor).count(pat=pat, **kwargs)

    _check_series_equal(result_fr, result_pd)


@regex_patterns
def test_count_regex(data, pat, str_accessor, fletcher_variant):
    _check_str_to_int("count", data, str_accessor, fletcher_variant, pat=pat)


@string_patterns
def test_text_endswith(data, pat, str_accessor, fletcher_variant):
    _check_str_to_bool("endswith", data, str_accessor, fletcher_variant, pat=pat)


def _check_extract(func, str_accessor, fletcher_variant, data, regex):

    if str_accessor == "str":
        pytest.skip(f"{func} is not yet dispatched to the ExtensionArray")
        return

    index = pd.Index(range(1, len(data) + 1))
    ser_fr = _fr_series_from_data(data, fletcher_variant, index=index)
    result_fr = getattr(getattr(ser_fr, str_accessor), func)(regex)
    assert isinstance(result_fr[0].dtype, fr.FletcherBaseDtype)

    ser_pd = pd.Series(data, index=index)
    result_pd = getattr(ser_pd.str, func)(regex)

    tm.assert_frame_equal(result_pd, result_fr.astype(object))


@pytest.mark.parametrize("regex", ["([0-9]+)", "([0-9]+)\\+([a-z]+)*"])
@pytest.mark.parametrize(
    "data", [["123+"], ["123+a"], ["123+a", "123+"], ["123+", "123+a"]]
)
def test_extract(str_accessor, fletcher_variant, data, regex):
    _check_extract("extract", str_accessor, fletcher_variant, data, regex)


@pytest.mark.parametrize("regex", ["([0-9]+)", "([0-9]+)\\+([a-z]+)*"])
@pytest.mark.parametrize(
    "data", [["123+"], ["123+a"], ["123+a", "123+"], ["123+", "123+a"]]
)
def test_extractall(str_accessor, fletcher_variant, data, regex):
    _check_extract("extractall", str_accessor, fletcher_variant, data, regex)


@string_patterns
def test_find(data, pat, str_accessor, fletcher_variant):
    _check_str_to_int("find", data, str_accessor, fletcher_variant, sub=pat)


@string_patterns
def test_findall(data, pat, str_accessor, fletcher_variant):
    _check_str_to_int("findall", data, str_accessor, fletcher_variant, pat=pat)


@settings(deadline=None)
@given(
    data=st.lists(st.one_of(st_text, st.none())),
    n=st.integers(min_value=0, max_value=10),
)
def test_get(data, n, str_accessor, fletcher_variant):
    _check_str_to_str("get", data, str_accessor, fletcher_variant, i=n)


@string_patterns
def test_index(data, pat, str_accessor, fletcher_variant):
    _check_str_to_int("index", data, str_accessor, fletcher_variant, sub=pat)


@settings(deadline=None)
@given(data=st.lists(st.one_of(st_text, st.none())))
def test_len(data, str_accessor, fletcher_variant):
    _check_str_to_int("len", data, str_accessor, fletcher_variant)


@settings(deadline=None)
@given(
    data=st.lists(st.one_of(st_text, st.none())),
    n=st.integers(min_value=0, max_value=50),
)
def test_ljust(data, n, str_accessor, fletcher_variant):
    _check_str_to_str("ljust", data, str_accessor, fletcher_variant, width=n)


@settings(deadline=None)
@given(data=st.lists(st.one_of(st_text, st.none())))
def test_lower(data, str_accessor, fletcher_variant):
    _check_str_to_str("lower", data, str_accessor, fletcher_variant)


@settings(deadline=None)
@given(data=st.lists(st.one_of(st_text, st.none())))
@strip_examples
def test_lstrip(str_accessor, fletcher_variant, data):
    _do_test_text_strip(str_accessor, fletcher_variant, 1, data, strip_method="lstrip")


@pytest.mark.parametrize("case", [True, False])
@pytest.mark.parametrize("pat", ["([0-9]+)", "([0-9]+)\\+([a-z]+)*"])
@pytest.mark.parametrize(
    "data", [["123+"], ["123+a"], ["123+a", "123+"], ["123+", "123+a"]]
)
def test_match(data, pat, case, str_accessor, fletcher_variant):
    _check_str_to_bool(
        "match", data, str_accessor, fletcher_variant, pat=pat, case=case
    )


@settings(deadline=None)
@given(data=st.lists(st.one_of(st_text, st.none())))
@pytest.mark.parametrize("form", ["NFC", "NFKC", "NFD", "NFKD"])
def test_normalize(data, form, str_accessor, fletcher_variant):
    _check_str_to_str("normalize", data, str_accessor, fletcher_variant, form=form)


@settings(deadline=None)
@given(
    data=st.lists(st.one_of(st_text, st.none())),
    n=st.integers(min_value=0, max_value=50),
)
@pytest.mark.parametrize("side", ["left", "right", "both"])
def test_pad(data, n, side, str_accessor, fletcher_variant):
    _check_str_to_str("pad", data, str_accessor, fletcher_variant, width=n, side=side)


@pytest.mark.parametrize("data", [["123"], ["123+"], ["123+a+", "123+"]])
@pytest.mark.parametrize("expand", [True, False])
def test_partition(str_accessor, fletcher_variant, data, expand):
    if not expand:
        pytest.xfail(
            "partition(expand=False) not supported as pyarrow cannot deal with tuples"
        )
    if str_accessor == "str":
        pytest.xfail("string.parititon always returns a tuple")
    _do_test_split(
        str_accessor, fletcher_variant, data, expand, split_method="partition"
    )


@settings(deadline=None)
@given(
    data=st.lists(st.one_of(st_text, st.none())),
    n=st.integers(min_value=0, max_value=10),
)
def test_repeat(data, n, str_accessor, fletcher_variant):
    _check_str_to_str("repeat", data, str_accessor, fletcher_variant, repeats=n)


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
def test_replace_no_regex_case_sensitive(
    data_tuple, repl, n, str_accessor, fletcher_variant
):
    data, pat, test_offset = data_tuple
    _check_str_to_str(
        "replace",
        data,
        str_accessor,
        fletcher_variant,
        test_offset=test_offset,
        pat=pat,
        repl=repl,
        n=n,
        case=True,
        regex=False,
    )


@string_patterns
def test_rfind(data, pat, str_accessor, fletcher_variant):
    _check_str_to_int("rfind", data, str_accessor, fletcher_variant, sub=pat)


@string_patterns
def test_rindex(data, pat, str_accessor, fletcher_variant):
    _check_str_to_int("index", data, str_accessor, fletcher_variant, sub=pat)


@settings(deadline=None)
@given(
    data=st.lists(st.one_of(st_text, st.none())),
    n=st.integers(min_value=0, max_value=50),
)
def test_rjust(data, n, str_accessor, fletcher_variant):
    _check_str_to_str("rjust", data, str_accessor, fletcher_variant, width=n)


@pytest.mark.parametrize("data", [["123"], ["123+"], ["123+a+", "123+"]])
@pytest.mark.parametrize("expand", [True, False])
def test_rpartition(str_accessor, fletcher_variant, data, expand):
    if not expand:
        pytest.xfail(
            "partition(expand=False) not supported as pyarrow cannot deal with tuples"
        )
    if str_accessor == "str":
        pytest.xfail("string.parititon always returns a tuple")
    _do_test_split(
        str_accessor, fletcher_variant, data, expand, split_method="rpartition"
    )


@settings(deadline=None)
@given(data=st.lists(st.one_of(st_text, st.none())))
@strip_examples
def test_rstrip(str_accessor, fletcher_variant, data):
    _do_test_text_strip(str_accessor, fletcher_variant, 1, data, strip_method="rstrip")


@settings(deadline=None)
@given(
    data=st.lists(st.one_of(st_text, st.none())),
    slice_=st.tuples(st.integers(-20, 20), st.integers(-20, 20), st.integers(-20, 20)),
)
def test_slice(data, slice_, str_accessor, fletcher_variant):
    if slice_[2] == 0:
        pytest.raises(ValueError)
        return
    if data == [None] or data == [""]:
        return

    ser_fr = _fr_series_from_data(data, fletcher_variant)
    result_fr = getattr(ser_fr, str_accessor).slice(*slice_)
    result_fr = result_fr.astype(object)
    # Pandas returns np.nan for NA values in cat, keep this in line
    result_fr[result_fr.isna()] = np.nan

    ser_pd = pd.Series(data, dtype=object)
    result_pd = ser_pd.str.slice(*slice_)

    tm.assert_series_equal(result_fr, result_pd)


def test_slice_replace(str_accessor, fletcher_variant):
    ser = _fr_series_from_data(["a", "ab", "abc", "abdc", "abcde"], fletcher_variant)

    # Using test cases from the pandas documentation
    result = getattr(ser, str_accessor).slice_replace(1, repl="X")
    expected = _fr_series_from_data(["aX", "aX", "aX", "aX", "aX"], fletcher_variant)
    tm.assert_series_equal(result, expected)

    result = getattr(ser, str_accessor).slice_replace(stop=2, repl="X")
    expected = _fr_series_from_data(["X", "X", "Xc", "Xdc", "Xcde"], fletcher_variant)
    tm.assert_series_equal(result, expected)

    result = getattr(ser, str_accessor).slice_replace(start=1, stop=3, repl="X")
    expected = _fr_series_from_data(["aX", "aX", "aX", "aXc", "aXde"], fletcher_variant)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("data", [["123"], ["123+"], ["123+a+", "123+"]])
@pytest.mark.parametrize("expand", [True, False])
def test_split(str_accessor, fletcher_variant, data, expand):
    _do_test_split(str_accessor, fletcher_variant, data, expand, split_method="split")


@pytest.mark.parametrize("data", [["123"], ["123+"], ["123+a+", "123+"]])
@pytest.mark.parametrize("expand", [True, False])
def test_rsplit(str_accessor, fletcher_variant, data, expand):
    _do_test_split(str_accessor, fletcher_variant, data, expand, split_method="rsplit")


def _do_test_split(str_accessor, fletcher_variant, data, expand, split_method):
    len_data = len(data)
    idx_a = list(range(1, len_data + 1))
    idx_b = list(range(2, len_data + 2))
    index = pd.MultiIndex.from_tuples(
        list(zip(idx_a, idx_b)), names=["first", "second"]
    )

    ser_fr = _fr_series_from_data(data, fletcher_variant, index=index)
    result_fr = getattr(getattr(ser_fr, str_accessor), split_method)("+", expand=expand)

    ser_pd = pd.Series(data, index=index)
    result_pd = getattr(ser_pd.str, split_method)("+", expand=expand)

    if expand:
        tm.assert_frame_equal(result_pd, result_fr.astype(object))
    else:
        tm.assert_series_equal(result_pd, result_fr.astype(object))


@string_patterns
def test_startswith(data, pat, str_accessor, fletcher_variant):
    _check_str_to_bool("startswith", data, str_accessor, fletcher_variant, pat=pat)


@settings(deadline=None, max_examples=3)
@given(data=st.lists(st.one_of(st_text, st.none())))
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
def test_strip_offset(str_accessor, fletcher_variant, fletcher_slice_offset, data):
    _do_test_text_strip(str_accessor, fletcher_variant, fletcher_slice_offset, data)


@settings(deadline=None)
@given(data=st.lists(st.one_of(st_text, st.none())))
@strip_examples
def test_strip(str_accessor, fletcher_variant, data):
    _do_test_text_strip(str_accessor, fletcher_variant, 1, data)


def _do_test_text_strip(
    str_accessor, fletcher_variant, fletcher_slice_offset, data, strip_method="strip"
):
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

    result_pd = getattr(ser_pd.str, strip_method)()
    result_fr = getattr(getattr(ser_fr, str_accessor), strip_method)()
    result_fr = result_fr.astype(object)
    # Pandas returns np.nan for NA values in cat, keep this in line
    result_fr[result_fr.isna()] = np.nan
    result_pd[result_pd.isna()] = np.nan
    tm.assert_series_equal(result_fr, result_pd)


@settings(deadline=None)
@given(data=st.lists(st.one_of(st_text, st.none())))
def test_swapcase(data, str_accessor, fletcher_variant):
    _check_str_to_str("swapcase", data, str_accessor, fletcher_variant)


@settings(deadline=None)
@given(data=st.lists(st.one_of(st_text, st.none())))
def test_title(data, str_accessor, fletcher_variant):
    _check_str_to_str("title", data, str_accessor, fletcher_variant)


@settings(deadline=None)
@given(data=st.lists(st.one_of(st_text, st.none())))
@example(data=["a"])
@example(data=["aa"])
@example(data=["1, 👅, 3"])
def test_translate(data, str_accessor, fletcher_variant):
    _check_str_to_str(
        "translate",
        data,
        str_accessor,
        fletcher_variant,
        table={"a": "🤙", "👅": "a", "1": "1"},
    )


@settings(deadline=None)
@given(data=st.lists(st.one_of(st_text, st.none())))
def test_upper(data, str_accessor, fletcher_variant):
    _check_str_to_str("upper", data, str_accessor, fletcher_variant)


def test_wrap(str_accessor, fletcher_variant):
    ser = _fr_series_from_data(
        ["line to be wrapped", "an-other line to\tbe wrapped"], fletcher_variant
    )

    result = getattr(ser, str_accessor).wrap(width=12)
    expected = _fr_series_from_data(
        ["line to be\nwrapped", "an-other\nline to\nbe wrapped"], fletcher_variant
    )
    tm.assert_series_equal(result, expected)

    result = getattr(ser, str_accessor).wrap(width=12, drop_whitespace=False)
    expected = _fr_series_from_data(
        ["line to be \nwrapped", "an-other \nline to\n        be \nwrapped"],
        fletcher_variant,
    )
    tm.assert_series_equal(result, expected)

    result = getattr(ser, str_accessor).wrap(width=5, break_long_words=True)
    expected = _fr_series_from_data(
        ["line\nto be\nwrapp\ned", "an-\nother\nline\nto\nbe wr\napped"],
        fletcher_variant,
    )
    tm.assert_series_equal(result, expected)

    result = getattr(ser, str_accessor).wrap(width=5, break_long_words=False)
    expected = _fr_series_from_data(
        ["line\nto be\nwrapped", "an-\nother\nline\nto\nbe\nwrapped"], fletcher_variant
    )
    tm.assert_series_equal(result, expected)

    result = getattr(ser, str_accessor).wrap(
        width=5, break_long_words=False, break_on_hyphens=False
    )
    expected = _fr_series_from_data(
        ["line\nto be\nwrapped", "an-other\nline\nto\nbe\nwrapped"], fletcher_variant
    )
    tm.assert_series_equal(result, expected)


def _optional_len(x: Optional[str]) -> int:
    if x is not None:
        return len(x)
    else:
        return 0


@settings(deadline=None)
@given(data=st.lists(st.one_of(st_text, st.none())))
def test_zfill(data, str_accessor, fletcher_variant):
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
    result_fr = getattr(ser_fr, str_accessor).zfill(max_str_len + 1)
    result_fr = result_fr.astype(object)
    # Pandas returns np.nan for NA values in cat, keep this in line
    result_fr[result_fr.isna()] = np.nan
    tm.assert_series_equal(result_fr, result_pd)


@settings(deadline=None)
@given(data=st.lists(st.one_of(st_text, st.none())))
def test_isalnum(data, str_accessor, fletcher_variant):
    data = filter_supported(data)
    _check_str_to_bool("isalnum", data, str_accessor, fletcher_variant)


@settings(deadline=None)
@given(data=st.lists(st.one_of(st_text, st.none())))
def test_isalpha(data, str_accessor, fletcher_variant):
    data = filter_supported(data)
    _check_str_to_bool("isalpha", data, str_accessor, fletcher_variant)


@settings(deadline=None)
@given(data=st.lists(st.one_of(st_text, st.none())))
def test_isdigit(data, str_accessor, fletcher_variant):
    data = filter_supported(data)
    _check_str_to_bool("isdigit", data, str_accessor, fletcher_variant)


@settings(deadline=None)
@given(data=st.lists(st.one_of(st_text, st.none())))
def test_isspace(data, str_accessor, fletcher_variant):
    data = filter_supported(data)
    _check_str_to_bool("isspace", data, str_accessor, fletcher_variant)


@settings(deadline=None)
# FIXME: Keep for now at max_codepoint 255 as we would otherwise need a lot of exclusions. Widen once we have a utf8proc release or we have dropped Python 3.6.
@given(
    data=st.lists(
        st.one_of(st.text(alphabet=st.characters(max_codepoint=255)), st.none())
    )
)
def test_islower(data, str_accessor, fletcher_variant):
    data = filter_supported(data)
    _check_str_to_bool("islower", data, str_accessor, fletcher_variant)


@settings(deadline=None)
# FIXME: Keep for now at max_codepoint 255 as we would otherwise need a lot of exclusions. Widen once we have a utf8proc release or we have dropped Python 3.6.
@given(
    data=st.lists(
        st.one_of(st.text(alphabet=st.characters(max_codepoint=255)), st.none())
    )
)
def test_isupper(data, str_accessor, fletcher_variant):
    data = filter_supported(data)
    _check_str_to_bool("isupper", data, str_accessor, fletcher_variant)


@settings(deadline=None)
@given(data=st.lists(st.one_of(st_text, st.none())))
def test_istitle(data, str_accessor, fletcher_variant):
    data = filter_supported(data)
    _check_str_to_bool("istitle", data, str_accessor, fletcher_variant)


@settings(deadline=None)
@given(data=st.lists(st.one_of(st_text, st.none())))
def test_isnumeric(data, str_accessor, fletcher_variant):
    data = filter_supported(data)
    _check_str_to_bool("isnumeric", data, str_accessor, fletcher_variant)


@settings(deadline=None)
@given(data=st.lists(st.one_of(st_text, st.none())))
def test_isdecimal(data, str_accessor, fletcher_variant):
    data = filter_supported(data)
    _check_str_to_bool("isdecimal", data, str_accessor, fletcher_variant)


def test_get_dummies(str_accessor, fletcher_variant):
    ser = _fr_series_from_data(["a|b", None, "a|c"], fletcher_variant)

    result = getattr(ser, str_accessor).get_dummies()
    if str_accessor == "str":
        expected = pd.DataFrame({"a": [1, 0, 1], "b": [1, 0, 0], "c": [0, 0, 1]})
    else:
        expected = pd.DataFrame(
            {
                "a": _fr_series_from_data(
                    [1, 0, 1], fletcher_variant, dtype=pa.int64()
                ),
                "b": _fr_series_from_data(
                    [1, 0, 0], fletcher_variant, dtype=pa.int64()
                ),
                "c": _fr_series_from_data(
                    [0, 0, 1], fletcher_variant, dtype=pa.int64()
                ),
            }
        )
    tm.assert_frame_equal(result, expected)


@numba.jit(nogil=True, nopython=True)
def prefix_length(s1, s1len, s2, s2len):
    min_len = min(s1len, s2len)
    prefix = 0
    for i in range(min_len):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return prefix


@pytest.mark.parametrize("parallel", [True, False])
def test_apply_binary_str_no_nulls_unchunked(parallel):
    a = pa.array(["a", "bb", "c"])
    b = pa.array(["aa", "bb", "dd"])
    expected = pa.array([1, 2, 0])

    result = apply_binary_str(
        a, b, func=prefix_length, output_dtype=np.int64, parallel=parallel
    )
    assert expected.equals(result)


@pytest.mark.parametrize("parallel", [True, False])
def test_apply_binary_str_nulls_unchunked(parallel):
    a = pa.array(["a", "bb", None, "c", "d"])
    b = pa.array(["aa", "bb", "dd", None, "c"])
    expected = pa.array([1, 2, None, None, 0])

    result = apply_binary_str(
        a, b, func=prefix_length, output_dtype=np.int64, parallel=parallel
    )
    assert expected.equals(result)


@pytest.mark.parametrize("parallel", [True, False])
def test_apply_binary_str_no_nulls_partly_chunked(parallel):
    a = pa.array(["a", "bb", "c"])
    b = pa.chunked_array([["aa"], ["bb", "dd"]])
    expected = pa.chunked_array([[1], [2, 0]])

    result = apply_binary_str(
        a, b, func=prefix_length, output_dtype=np.int64, parallel=parallel
    )
    assert expected.equals(result)


@pytest.mark.parametrize("parallel", [True, False])
def test_apply_binary_str_nulls_partly_chunked(parallel):
    a = pa.chunked_array([["a", "bb"], [None, "c", "d"]])
    b = pa.array(["aa", "bb", "dd", None, "c"])
    expected = pa.chunked_array([[1, 2], [None, None, 0]])

    result = apply_binary_str(
        a, b, func=prefix_length, output_dtype=np.int64, parallel=parallel
    )
    assert expected.equals(result)


@pytest.mark.parametrize("parallel", [True, False])
def test_apply_binary_str_no_nulls_chunked(parallel):
    a = pa.chunked_array([["a", "bb"], ["c"]])
    b = pa.chunked_array([["aa"], ["bb", "dd"]])
    expected = pa.chunked_array([[1], [2], [0]])

    result = apply_binary_str(
        a, b, func=prefix_length, output_dtype=np.int64, parallel=parallel
    )
    assert expected.equals(result)


@pytest.mark.parametrize("parallel", [True, False])
def test_apply_binary_str_nulls_chunked(parallel):
    a = pa.chunked_array([["a", "bb"], [None, "c", "d"]])
    b = pa.chunked_array([["aa", "bb", "dd"], [None, "c"]])
    expected = pa.chunked_array([[1, 2, None, None, 0]])

    result = apply_binary_str(
        a, b, func=prefix_length, output_dtype=np.int64, parallel=parallel
    )
    assert expected.equals(result)
