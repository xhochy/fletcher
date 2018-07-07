# -*- coding: utf-8 -*-

import datetime
import pandas as pd
import pyarrow as pa
import pytest
import six
import string
import sys
from collections import namedtuple
from distutils.version import LooseVersion
from pandas.tests.extension.base import (
    BaseCastingTests,
    BaseConstructorsTests,
    BaseDtypeTests,
    BaseGetitemTests,
    BaseGroupbyTests,
    BaseInterfaceTests,
    BaseMethodsTests,
    BaseMissingTests,
    BaseReshapingTests,
    BaseSetitemTests,
)

from fletcher import FletcherArray, FletcherDtype

FletcherTestType = namedtuple(
    "FletcherTestType",
    [
        "dtype",
        "data",
        "data_missing",
        "data_for_grouping",
        "data_for_sorting",
        "data_missing_for_sorting",
        "data_repeated",
    ],
)

if sys.version_info >= (3, 6):
    from random import choices
else:
    from random import choice

    def choices(seq, k):
        return [choice(seq) for i in range(k)]


test_types = [
    FletcherTestType(
        pa.string(),
        [u"🙈", u"Ö", u"Č", u"a", u"B"] * 20,
        [None, "A"],
        ["B", "B", None, None, "A", "A", "B", "C"],
        ["B", "C", "A"],
        ["B", None, "A"],
        lambda: choices(list(string.ascii_letters), k=10),
    ),
    # Float and int types require the support of constant memoryviews, this
    # depends on https://github.com/pandas-dev/pandas/pull/21688
    pytest.param(
        FletcherTestType(
            pa.int64(),
            [2, 1, -1, 0, 66] * 20,
            [None, 1],
            [2, 2, None, None, -100, -100, 2, 100],
            [2, 100, -10],
            [2, None, -10],
            lambda: choices(list(range(100)), k=10),
        ),
        marks=pytest.mark.xfail,
    ),
    pytest.param(
        FletcherTestType(
            pa.float64(),
            [2.5, 1.0, -1.0, 0, 66.6] * 20,
            [None, 1.1],
            [2.5, 2.5, None, None, -100.1, -100.1, 2.5, 100.1],
            [2.5, 100.99, -10.1],
            [2.5, None, -10.1],
            lambda: choices([2.5, 1.0, -1.0, 0, 66.6], k=10),
        ),
        marks=pytest.mark.xfail,
    ),
    # Most of the tests fail as assert_extension_array_equal casts to numpy object
    # arrays and on them equality is not defined.
    pytest.param(
        FletcherTestType(
            pa.list_(pa.string()),
            [["B", "C"], ["A"], [None], ["A", "A"], []],
            [None, ["A"]],
            [["B"], ["B"], None, None, ["A"], ["A"], ["B"], ["C"]],
            [["B"], ["C"], ["A"]],
            [["B"], None, ["A"]],
            lambda: choices([["B", "C"], ["A"], [None], ["A", "A"]], k=10),
        ),
        marks=pytest.mark.xfail,
    ),
    FletcherTestType(
        pa.date64(),
        [
            datetime.date(2015, 1, 1),
            datetime.date(2010, 12, 31),
            datetime.date(1970, 1, 1),
            datetime.date(1900, 3, 31),
            datetime.date(1999, 12, 31),
        ]
        * 20,
        [None, datetime.date(2015, 1, 1)],
        [
            datetime.date(2015, 2, 2),
            datetime.date(2015, 2, 2),
            None,
            None,
            datetime.date(2015, 1, 1),
            datetime.date(2015, 1, 1),
            datetime.date(2015, 2, 2),
            datetime.date(2015, 3, 3),
        ],
        [
            datetime.date(2015, 2, 2),
            datetime.date(2015, 3, 3),
            datetime.date(2015, 1, 1),
        ],
        [datetime.date(2015, 2, 2), None, datetime.date(2015, 1, 1)],
        lambda: choices(list(pd.date_range("2010-1-1", "2011-1-1").date), k=10),
    ),
]


@pytest.fixture(params=test_types)
def fletcher_type(request):
    return request.param


@pytest.fixture
def dtype(fletcher_type):
    return FletcherDtype(fletcher_type.dtype)


@pytest.fixture
def data(fletcher_type):
    return FletcherArray(fletcher_type.data, dtype=fletcher_type.dtype)


@pytest.fixture
def data_missing(fletcher_type):
    return FletcherArray(fletcher_type.data_missing, dtype=fletcher_type.dtype)


@pytest.fixture
def data_repeated(fletcher_type):
    """Return different versions of data for count times"""

    def gen(count):
        for _ in range(count):
            yield FletcherArray(
                fletcher_type.data_repeated(), dtype=fletcher_type.dtype
            )

    yield gen


@pytest.fixture
def data_for_grouping(fletcher_type):
    """Data for factorization, grouping, and unique tests.

    Expected to be like [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA is missing
    """
    return FletcherArray(fletcher_type.data_for_grouping, dtype=fletcher_type.dtype)


@pytest.fixture
def data_for_sorting(fletcher_type):
    """Length-3 array with a known sort order.

    This should be three items [B, C, A] with
    A < B < C
    """
    return FletcherArray(fletcher_type.data_for_sorting, dtype=fletcher_type.dtype)


@pytest.fixture
def data_missing_for_sorting(fletcher_type):
    """Length-3 array with a known sort order.

    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    return FletcherArray(
        fletcher_type.data_missing_for_sorting, dtype=fletcher_type.dtype
    )


class TestBaseCasting(BaseCastingTests):

    @pytest.mark.xfail(six.PY2, reason="Cast of UTF8 to `str` fails in py2.")
    def test_astype_str(self, data):
        BaseCastingTests.test_astype_str(self, data)


class TestBaseConstructors(BaseConstructorsTests):

    @pytest.mark.xfail(reason="Tries to construct dtypes with np.dtype")
    def test_from_dtype(self, data):
        if pa.types.is_string(data.dtype.arrow_dtype):
            pytest.xfail(
                "String construction is failing as Pandas wants to pass the FletcherDtype to NumPy"
            )
        BaseConstructorsTests.test_from_dtype(self, data)


class TestBaseDtype(BaseDtypeTests):
    pass


class TestBaseGetitemTests(BaseGetitemTests):

    def test_take_non_na_fill_value(self, data_missing):
        if pa.types.is_integer(data_missing.dtype.arrow_dtype):
            pytest.mark.xfail(reasion="Take is not yet correctly implemented for ints")
        else:
            BaseGetitemTests.test_take_non_na_fill_value(self, data_missing)

    def test_reindex_non_na_fill_value(self, data_missing):
        if pa.types.is_integer(data_missing.dtype.arrow_dtype):
            pytest.mark.xfail(reasion="Take is not yet correctly implemented for ints")
        else:
            BaseGetitemTests.test_reindex_non_na_fill_value(self, data_missing)

    @pytest.mark.skip
    def test_reindex(self):
        # No longer available in master and fails with pandas 0.23.1
        # due to a dtype assumption that does not hold for Arrow
        pass


class TestBaseGroupbyTests(BaseGroupbyTests):
    pass


class TestBaseInterfaceTests(BaseInterfaceTests):
    pass


class TestBaseMethodsTests(BaseMethodsTests):

    @pytest.mark.parametrize("dropna", [True, False])
    def test_value_counts(self, all_data, dropna, dtype):
        if LooseVersion(pd.__version__) >= "0.24.0dev0":
            pytest.skip("Master requires value_counts but not part of the interface")
        # Skip integer tests while there is no support for ExtensionIndex.
        # The dropna=True variant will produce a mix of IntIndex and FloatIndex.
        if dtype.name == "fletcher[int64]":
            pytest.skip("ExtensionIndex is no yet implemented")
        else:
            BaseMethodsTests.test_value_counts(self, all_data, dropna)

    def test_combine_le(self, data_repeated):
        if LooseVersion(pd.__version__) <= "0.24.0dev0":
            pytest.skip("Test only exists on master")
        # GH 20825
        # Test that combine works when doing a <= (le) comparison
        # Fletcher returns 'fletcher[bool]' instead of np.bool as dtype
        orig_data1, orig_data2 = data_repeated(2)
        s1 = pd.Series(orig_data1)
        s2 = pd.Series(orig_data2)
        result = s1.combine(s2, lambda x1, x2: x1 <= x2)
        expected = pd.Series(
            orig_data1._from_sequence(
                [a <= b for (a, b) in zip(list(orig_data1), list(orig_data2))]
            )
        )
        self.assert_series_equal(result, expected)

        val = s1.iloc[0]
        result = s1.combine(val, lambda x1, x2: x1 <= x2)
        expected = pd.Series(
            orig_data1._from_sequence([a <= val for a in list(orig_data1)])
        )
        self.assert_series_equal(result, expected)

    def test_combine_add(self, data_repeated, dtype):
        if LooseVersion(pd.__version__) <= "0.24.0dev0":
            pytest.skip("Test only exists on master")
        if dtype.name == "fletcher[date64[ms]]":
            pytest.skip(
                "unsupported operand type(s) for +: 'datetime.date' and 'datetime.date"
            )
        else:
            BaseMethodsTests.test_combine_add(self, data_repeated)


class TestBaseMissingTests(BaseMissingTests):
    pass


class TestBaseReshapingTests(BaseReshapingTests):

    def test_concat_mixed_dtypes(self, data, dtype):
        if dtype.name in ["fletcher[int64]", "fletcher[double]"]:
            # https://github.com/pandas-dev/pandas/issues/21792
            pytest.skip("pd.concat(int64, fletcher[int64] yields int64")
        else:
            BaseReshapingTests.test_concat_mixed_dtypes(self, data)


class TestBaseSetitemTests(BaseSetitemTests):
    pass
