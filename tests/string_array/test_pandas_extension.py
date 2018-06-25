# -*- coding: utf-8 -*-

import datetime
import pyarrow as pa
import pytest
import six
from collections import namedtuple
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
    ],
)

test_types = [
    FletcherTestType(
        pa.string(),
        [u"🙈", u"Ö", u"Č", u"a", u"B"] * 20,
        [None, "A"],
        ["B", "B", None, None, "A", "A", "B", "C"],
        ["B", "C", "A"],
        ["B", None, "A"],
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
    pass


class TestBaseDtype(BaseDtypeTests):
    pass


class TestBaseGetitemTests(BaseGetitemTests):

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
    pass


@pytest.mark.xfail()
class TestBaseMissingTests(BaseMissingTests):
    pass


@pytest.mark.xfail()
class TestBaseReshapingTests(BaseReshapingTests):
    pass


@pytest.mark.xfail()
class TestBaseSetitemTests(BaseSetitemTests):
    pass
