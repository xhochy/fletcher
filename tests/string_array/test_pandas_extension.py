# -*- coding: utf-8 -*-

import pytest
import six
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

from fletcher import StringArray, StringDtype


@pytest.fixture
def dtype():
    return StringDtype()


@pytest.fixture
def data():
    candidates = [u"🙈", u"Ö", u"Č", u"a", u"B"]
    return StringArray(candidates * 20)


@pytest.fixture
def data_missing():
    return StringArray([None, "A"])


@pytest.fixture
def data_for_grouping():
    """Data for factorization, grouping, and unique tests.

    Expected to be like [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA is missing
    """
    return StringArray(["B", "B", None, None, "A", "A", "B", "C"])


@pytest.fixture
def data_for_sorting():
    """Length-3 array with a known sort order.

    This should be three items [B, C, A] with
    A < B < C
    """
    return StringArray(["B", "C", "A"])


@pytest.fixture
def data_missing_for_sorting():
    """Length-3 array with a known sort order.

    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    return StringArray(["B", None, "A"])


class TestBaseCasting(BaseCastingTests):

    @pytest.mark.xfail(six.PY2, reason="Cast of UTF8 to `str` fails in py2.")
    def test_astype_str(self, data):
        BaseCastingTests.test_astype_str(self, data)


class TestBaseConstructors(BaseConstructorsTests):
    pass


class TestBaseDtype(BaseDtypeTests):
    pass


class TestBaseGetitemTests(BaseGetitemTests):
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
