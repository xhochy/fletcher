# -*- coding: utf-8 -*-

import pytest
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

from fletcher import StringDtype, StringArray
from random import choice


@pytest.fixture
def dtype():
    return StringDtype()


@pytest.fixture
def data():
    candidates = ["a", "Ö", "Č", "🙈"]
    return StringArray([choice(candidates) for x in range(100)])


@pytest.fixture
def data_missing():
    return StringArray(["A", None])


@pytest.fixture
def data_for_grouping():
    """Data for factorization, grouping, and unique tests.

    Expected to be like [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA is missing
    """
    raise StringArray(["B", "B", None, None, "A", "A", "B", "C"])


@pytest.fixture
def data_for_sorting():
    """Length-3 array with a known sort order.

    This should be three items [B, C, A] with
    A < B < C
    """
    raise StringArray(["B", "C", "A"])


@pytest.fixture
def data_missing_for_sorting():
    """Length-3 array with a known sort order.

    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    raise StringArray(["B", None, "A"])


@pytest.mark.xfail()
class TestBaseCasting(BaseCastingTests):
    pass


@pytest.mark.xfail()
class TestBaseConstructors(BaseConstructorsTests):
    pass


@pytest.mark.xfail()
class TestBaseDtype(BaseDtypeTests):
    pass


@pytest.mark.xfail()
class TestBaseGetitemTests(BaseGetitemTests):
    pass


@pytest.mark.xfail()
class TestBaseGroupbyTests(BaseGroupbyTests):
    pass


@pytest.mark.xfail()
class TestBaseInterfaceTests(BaseInterfaceTests):
    pass


@pytest.mark.xfail()
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
