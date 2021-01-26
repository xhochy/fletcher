import datetime
import string
from collections import namedtuple
from distutils.version import LooseVersion
from random import choices
from typing import Optional, Type

import numpy as np
import pandas as pd
import pandas.util.testing as pdt
import pyarrow as pa
import pytest
from pandas.tests.extension.base import (
    BaseArithmeticOpsTests,
    BaseBooleanReduceTests,
    BaseCastingTests,
    BaseComparisonOpsTests,
    BaseConstructorsTests,
    BaseDtypeTests,
    BaseGetitemTests,
    BaseGroupbyTests,
    BaseInterfaceTests,
    BaseMethodsTests,
    BaseMissingTests,
    BaseNoReduceTests,
    BaseNumericReduceTests,
    BaseParsingTests,
    BasePrintingTests,
    BaseReshapingTests,
    BaseSetitemTests,
)

from fletcher import FletcherBaseDtype

if LooseVersion(pd.__version__) >= "0.25.0":
    # imports of pytest fixtures needed for derived unittest classes
    from pandas.tests.extension.conftest import as_array  # noqa: F401; noqa: F401
    from pandas.tests.extension.conftest import as_frame  # noqa: F401
    from pandas.tests.extension.conftest import as_series  # noqa: F401
    from pandas.tests.extension.conftest import groupby_apply_op  # noqa: F401
    from pandas.tests.extension.conftest import use_numpy  # noqa: F401


PANDAS_GE_1_1_0 = LooseVersion(pd.__version__) >= "1.1.0"


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


def is_arithmetic_type(arrow_dtype: pa.DataType) -> bool:
    """Check whether this is a type that support arithmetics."""
    return (
        pa.types.is_integer(arrow_dtype)
        or pa.types.is_floating(arrow_dtype)
        or pa.types.is_decimal(arrow_dtype)
    )


skip_non_artithmetic_type = pytest.mark.skip_by_type_filter(
    [lambda x: not is_arithmetic_type(x)]
)
xfail_list_scalar_constuctor_not_implemented = pytest.mark.xfail_by_type_filter(
    [pa.types.is_list], "constructor from scalars is not implemented for lists"
)
xfail_list_equals_not_implemented = pytest.mark.xfail_by_type_filter(
    [pa.types.is_list], "== is not implemented for lists"
)
xfail_list_setitem_not_implemented = pytest.mark.xfail_by_type_filter(
    [pa.types.is_list], "__setitem__ is not implemented for lists"
)
xfail_missing_list_dict_encode = pytest.mark.xfail_by_type_filter(
    [pa.types.is_list],
    "ArrowNotImplementedError: dictionary-encode not implemented for list<item: string>",
)
xfail_bool_too_few_uniques = pytest.mark.xfail_by_type_filter(
    [pa.types.is_boolean], "Test requires at least 3 unique values"
)


test_types = [
    FletcherTestType(
        pa.string(),
        ["🙈", "Ö", "Č", "a", "B"] * 20,
        [None, "A"],
        ["B", "B", None, None, "A", "A", "B", "C"],
        ["B", "C", "A"],
        ["B", None, "A"],
        lambda: choices(list(string.ascii_letters), k=10),
    ),
    FletcherTestType(
        pa.bool_(),
        [True, False, True, True, False] * 20,
        [None, False],
        [True, True, None, None, False, False, True, False],
        [True, False, False],
        [True, None, False],
        lambda: choices([True, False], k=10),
    ),
    FletcherTestType(
        pa.int8(),
        # Use small values here so that np.prod stays in int32
        [2, 1, 1, 2, 1] * 20,
        [None, 1],
        [2, 2, None, None, -100, -100, 2, 100],
        [2, 100, -10],
        [2, None, -10],
        lambda: choices(list(range(100)), k=10),
    ),
    FletcherTestType(
        pa.int16(),
        # Use small values here so that np.prod stays in int32
        [2, 1, 3, 2, 1] * 20,
        [None, 1],
        [2, 2, None, None, -100, -100, 2, 100],
        [2, 100, -10],
        [2, None, -10],
        lambda: choices(list(range(100)), k=10),
    ),
    FletcherTestType(
        pa.int32(),
        # Use small values here so that np.prod stays in int32
        [2, 1, 3, 2, 1] * 20,
        [None, 1],
        [2, 2, None, None, -100, -100, 2, 100],
        [2, 100, -10],
        [2, None, -10],
        lambda: choices(list(range(100)), k=10),
    ),
    FletcherTestType(
        pa.int64(),
        # Use small values here so that np.prod stays in int64
        [2, 1, 3, 2, 1] * 20,
        [None, 1],
        [2, 2, None, None, -100, -100, 2, 100],
        [2, 100, -10],
        [2, None, -10],
        lambda: choices(list(range(100)), k=10),
    ),
    FletcherTestType(
        pa.float64(),
        [2, 1.0, 1.0, 5.5, 6.6] * 20,
        [None, 1.1],
        [2.5, 2.5, None, None, -100.1, -100.1, 2.5, 100.1],
        [2.5, 100.99, -10.1],
        [2.5, None, -10.1],
        lambda: choices([2.5, 1.0, -1.0, 0, 66.6], k=10),
    ),
    # Most of the tests fail as assert_extension_array_equal casts to numpy object
    # arrays and on them equality is not defined.
    pytest.param(
        FletcherTestType(
            pa.list_(pa.string()),
            [["B", "C"], ["A"], [None], ["A", "A"], []] * 20,
            [None, ["A"]],
            [["B"], ["B"], None, None, ["A"], ["A"], ["B"], ["C"]],
            [["B"], ["C"], ["A"]],
            [["B"], None, ["A"]],
            lambda: choices([["B", "C"], ["A"], [None], ["A", "A"]], k=10),
        )
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


@pytest.fixture(params=[True, False])
def box_in_series(request):
    """Whether to box the data in a Series."""
    return request.param


@pytest.fixture(params=test_types)
def fletcher_type(request):
    return request.param


@pytest.fixture(autouse=True)
def skip_by_type_filter(request, fletcher_type):
    if request.node.get_closest_marker("skip_by_type_filter"):
        for marker in request.node.iter_markers("skip_by_type_filter"):
            for func in marker.args[0]:
                if func(fletcher_type.dtype):
                    pytest.skip(f"skipped for type: {fletcher_type}")


@pytest.fixture(autouse=True)
def xfail_by_type_filter(request, fletcher_type):
    if request.node.get_closest_marker("xfail_by_type_filter"):
        for marker in request.node.iter_markers("xfail_by_type_filter"):
            for func in marker.args[0]:
                if func(fletcher_type.dtype):
                    pytest.xfail(f"XFAIL for type: {fletcher_type}")


@pytest.fixture
def dtype(fletcher_type, fletcher_dtype):
    return fletcher_dtype(fletcher_type.dtype)


@pytest.fixture
def data(fletcher_type, fletcher_array):
    return fletcher_array(fletcher_type.data, dtype=fletcher_type.dtype)


@pytest.fixture
def data_for_twos(dtype, fletcher_type, fletcher_array):
    if dtype._is_numeric:
        return fletcher_array([2] * 100, dtype=fletcher_type.dtype)
    else:
        return None


@pytest.fixture
def data_missing(fletcher_type, fletcher_array):
    return fletcher_array(fletcher_type.data_missing, dtype=fletcher_type.dtype)


@pytest.fixture
def data_repeated(fletcher_type, fletcher_array):
    """Return different versions of data for count times."""
    pass  # noqa

    def gen(count):
        for _ in range(count):
            yield fletcher_array(
                fletcher_type.data_repeated(), dtype=fletcher_type.dtype
            )

    yield gen


@pytest.fixture
def data_for_grouping(fletcher_type, fletcher_array):
    """Fixture with data for factorization, grouping, and unique tests.

    Expected to be like [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA is missing
    """
    return fletcher_array(fletcher_type.data_for_grouping, dtype=fletcher_type.dtype)


@pytest.fixture
def data_for_sorting(fletcher_type, fletcher_array):
    """Length-3 array with a known sort order.

    This should be three items [B, C, A] with
    A < B < C
    """
    return fletcher_array(fletcher_type.data_for_sorting, dtype=fletcher_type.dtype)


@pytest.fixture
def data_missing_for_sorting(fletcher_type, fletcher_array):
    """Length-3 array with a known sort order.

    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    return fletcher_array(
        fletcher_type.data_missing_for_sorting, dtype=fletcher_type.dtype
    )


@pytest.fixture(params=[None, lambda x: x])
def sort_by_key(request):
    """
    Return a simple fixture for festing keys in sorting methods.

    Tests None (no key) and the identity key.
    """
    return request.param


@pytest.fixture(params=[None, np.nan, pd.NaT, float("nan"), pd.NA], ids=str)
def nulls_fixture(request):
    """
    Fixture for each null type in pandas.
    """
    return request.param


def get_dtype(obj):
    if hasattr(pdt, "get_dtype"):
        return pdt.get_dtype(obj)
    else:
        if isinstance(obj, pd.DataFrame):
            return obj.dtypes.iat[0]
        return obj.dtype


class TestBaseCasting(BaseCastingTests):
    pass


class TestBaseConstructors(BaseConstructorsTests):
    def test_from_dtype(self, data):
        if pa.types.is_string(data.dtype.arrow_dtype):
            pytest.xfail(
                "String construction is failing as Pandas wants to pass the FletcherChunkedDtype to NumPy"
            )
        BaseConstructorsTests.test_from_dtype(self, data)

    @xfail_list_scalar_constuctor_not_implemented
    def test_series_constructor_scalar_with_index(self, data, dtype):
        if PANDAS_GE_1_1_0:
            BaseConstructorsTests.test_series_constructor_scalar_with_index(
                self, data, dtype
            )


class TestBaseDtype(BaseDtypeTests):
    pass


class TestBaseGetitemTests(BaseGetitemTests):
    def test_loc_iloc_frame_single_dtype(self, data):
        if pa.types.is_string(data.dtype.arrow_dtype):
            pytest.mark.xfail(
                reason="https://github.com/pandas-dev/pandas/issues/27673"
            )
        else:
            BaseGetitemTests.test_loc_iloc_frame_single_dtype(self, data)


class TestBaseGroupbyTests(BaseGroupbyTests):
    @xfail_bool_too_few_uniques
    @xfail_missing_list_dict_encode
    @pytest.mark.parametrize("as_index", [True, False])
    def test_groupby_extension_agg(self, as_index, data_for_grouping):
        BaseGroupbyTests.test_groupby_extension_agg(self, as_index, data_for_grouping)

    @xfail_bool_too_few_uniques
    @xfail_missing_list_dict_encode
    def test_groupby_extension_no_sort(self, data_for_grouping):
        BaseGroupbyTests.test_groupby_extension_no_sort(self, data_for_grouping)

    @xfail_missing_list_dict_encode
    def test_groupby_extension_transform(self, data_for_grouping):
        if pa.types.is_boolean(data_for_grouping.dtype.arrow_dtype):
            valid = data_for_grouping[~data_for_grouping.isna()]
            df = pd.DataFrame({"A": [1, 1, 3, 3, 1, 4], "B": valid})

            result = df.groupby("B").A.transform(len)
            # Expected grouping is different as we only have two non-null values
            expected = pd.Series([3, 3, 3, 3, 3, 3], name="A")

            self.assert_series_equal(result, expected)
        else:
            BaseGroupbyTests.test_groupby_extension_transform(self, data_for_grouping)

    @xfail_missing_list_dict_encode
    def test_groupby_extension_apply(
        self, data_for_grouping, groupby_apply_op  # noqa: F811
    ):
        BaseGroupbyTests.test_groupby_extension_apply(
            self, data_for_grouping, groupby_apply_op
        )


class TestBaseInterfaceTests(BaseInterfaceTests):
    @pytest.mark.xfail(
        reason="view or self[:] returns a shallow copy in-place edits are not backpropagated"
    )
    def test_view(self, data):
        BaseInterfaceTests.test_view(self, data)

    def test_array_interface(self, data):
        if pa.types.is_list(data.dtype.arrow_dtype):
            pytest.xfail("Not sure whether this test really holds for list")
        else:
            BaseInterfaceTests.test_array_interface(self, data)

    @xfail_list_setitem_not_implemented
    def test_copy(self, data):
        BaseInterfaceTests.test_array_interface(self, data)

    @xfail_list_equals_not_implemented
    @pytest.mark.xfail_by_type_filter(
        [pa.types.is_date], "NaT and null are handled slightly differently"
    )
    @pytest.mark.xfail_by_type_filter(
        [pa.types.is_floating], "NaN and null are handled slightly differently"
    )
    def test_contains(self, data, data_missing):
        if hasattr(BaseInterfaceTests, "test_contains"):
            BaseInterfaceTests.test_contains(self, data, data_missing)


class TestBaseMethodsTests(BaseMethodsTests):

    # https://github.com/pandas-dev/pandas/issues/22843
    @pytest.mark.skip(reason="Incorrect expected")
    @pytest.mark.parametrize("dropna", [True, False])
    def test_value_counts(self, all_data, dropna, dtype):
        pass

    @xfail_list_equals_not_implemented
    @pytest.mark.parametrize("box", [pd.array, pd.Series, pd.DataFrame])
    def test_equals(self, data, na_value, as_series, box):  # noqa: F811
        if PANDAS_GE_1_1_0:
            BaseMethodsTests.test_equals(self, data, na_value, as_series, box)

    @xfail_missing_list_dict_encode
    def test_value_counts_with_normalize(self, data):
        if PANDAS_GE_1_1_0:
            BaseMethodsTests.test_value_counts_with_normalize(self, data)

    def test_combine_le(self, data_repeated):
        # GH 20825
        # Test that combine works when doing a <= (le) comparison
        # Fletcher returns 'fletcher_chunked[bool]' instead of np.bool as dtype
        orig_data1, orig_data2 = data_repeated(2)
        if pa.types.is_list(orig_data1.dtype.arrow_dtype):
            return pytest.skip("__le__ not implemented for list scalars with None")
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
        if dtype.name in [
            "fletcher_chunked[date64[ms]]",
            "fletcher_continuous[date64[ms]]",
        ]:
            pytest.skip(
                "unsupported operand type(s) for +: 'datetime.date' and 'datetime.date"
            )
        else:
            BaseMethodsTests.test_combine_add(self, data_repeated)

    @xfail_bool_too_few_uniques
    def test_argsort(self, data_for_sorting):
        BaseMethodsTests.test_argsort(self, data_for_sorting)

    @xfail_bool_too_few_uniques
    def test_argmin_argmax(self, data_for_sorting, data_missing_for_sorting, na_value):
        if PANDAS_GE_1_1_0:
            BaseMethodsTests.test_argmin_argmax(
                self, data_for_sorting, data_missing_for_sorting, na_value
            )
        else:
            pass

    @pytest.mark.parametrize("ascending", [True, False])
    @xfail_bool_too_few_uniques
    @xfail_missing_list_dict_encode
    def test_sort_values(self, data_for_sorting, ascending, sort_by_key):
        if PANDAS_GE_1_1_0:
            BaseMethodsTests.test_sort_values(
                self, data_for_sorting, ascending, sort_by_key
            )
        else:
            BaseMethodsTests.test_sort_values(self, data_for_sorting, ascending)

    @pytest.mark.parametrize("na_sentinel", [-1, -2])
    @xfail_bool_too_few_uniques
    @xfail_missing_list_dict_encode
    def test_factorize(self, data_for_grouping, na_sentinel):
        BaseMethodsTests.test_factorize(self, data_for_grouping, na_sentinel)

    @pytest.mark.parametrize("na_sentinel", [-1, -2])
    @xfail_bool_too_few_uniques
    @xfail_missing_list_dict_encode
    def test_factorize_equivalence(self, data_for_grouping, na_sentinel):
        BaseMethodsTests.test_factorize_equivalence(
            self, data_for_grouping, na_sentinel
        )

    @pytest.mark.parametrize("ascending", [True, False])
    @xfail_missing_list_dict_encode
    def test_sort_values_frame(self, data_for_sorting, ascending):
        BaseMethodsTests.test_sort_values_frame(self, data_for_sorting, ascending)

    @xfail_bool_too_few_uniques
    def test_searchsorted(self, data_for_sorting, as_series):  # noqa: F811
        BaseMethodsTests.test_searchsorted(self, data_for_sorting, as_series)

    @pytest.mark.parametrize("box", [pd.Series, lambda x: x])
    @pytest.mark.parametrize("method", [lambda x: x.unique(), pd.unique])
    @xfail_missing_list_dict_encode
    def test_unique(self, data, box, method):
        BaseMethodsTests.test_unique(self, data, box, method)

    @xfail_missing_list_dict_encode
    def test_factorize_empty(self, data):
        BaseMethodsTests.test_factorize_empty(self, data)

    def test_fillna_copy_frame(self, data_missing):
        if pa.types.is_list(data_missing.dtype.arrow_dtype):
            pytest.xfail("pandas' fillna cannot cope with lists as a scalar")
        else:
            BaseMethodsTests.test_fillna_copy_frame(self, data_missing)

    def test_fillna_copy_series(self, data_missing):
        if pa.types.is_list(data_missing.dtype.arrow_dtype):
            pytest.xfail("pandas' fillna cannot cope with lists as a scalar")
        else:
            BaseMethodsTests.test_fillna_copy_series(self, data_missing)

    @xfail_list_setitem_not_implemented
    def test_combine_first(self, data):
        BaseMethodsTests.test_combine_first(self, data)

    @xfail_list_setitem_not_implemented
    def test_shift_0_periods(self, data):
        if PANDAS_GE_1_1_0:
            BaseMethodsTests.test_shift_0_periods(self, data)

    def test_shift_fill_value(self, data):
        if pa.types.is_list(data.dtype.arrow_dtype):
            pytest.xfail("pandas' isna cannot cope with lists")
        else:
            BaseMethodsTests.test_shift_fill_value(self, data)

    def test_hash_pandas_object_works(self, data, as_frame):  # noqa: F811
        if pa.types.is_list(data.dtype.arrow_dtype):
            pytest.xfail("Fails on hashing ndarrays")
        else:
            BaseMethodsTests.test_hash_pandas_object_works(self, data, as_frame)

    @xfail_list_setitem_not_implemented
    def test_where_series(self, data, na_value, as_frame):  # noqa: F811
        BaseMethodsTests.test_where_series(self, data, na_value, as_frame)


class TestBaseMissingTests(BaseMissingTests):
    @pytest.mark.parametrize("method", ["ffill", "bfill"])
    def test_fillna_series_method(self, data_missing, method):
        BaseMissingTests.test_fillna_series_method(self, data_missing, method)

    def test_fillna_frame(self, data_missing):
        if pa.types.is_list(data_missing.dtype.arrow_dtype):
            pytest.xfail("pandas' fillna cannot cope with lists as a scalar")
        else:
            BaseMissingTests.test_fillna_frame(self, data_missing)

    def test_fillna_scalar(self, data_missing):
        if pa.types.is_list(data_missing.dtype.arrow_dtype):
            pytest.xfail("pandas' fillna cannot cope with lists as a scalar")
        else:
            BaseMissingTests.test_fillna_scalar(self, data_missing)

    def test_fillna_series(self, data_missing):
        if pa.types.is_list(data_missing.dtype.arrow_dtype):
            pytest.xfail("pandas' fillna cannot cope with lists as a scalar")
        else:
            BaseMissingTests.test_fillna_series(self, data_missing)


class TestBaseReshapingTests(BaseReshapingTests):
    def test_concat_mixed_dtypes(self, data, dtype):
        arrow_dtype = data.dtype.arrow_dtype
        if (
            pa.types.is_integer(arrow_dtype)
            or pa.types.is_floating(arrow_dtype)
            or pa.types.is_boolean(arrow_dtype)
        ):
            # https://github.com/pandas-dev/pandas/issues/21792
            pytest.skip("pd.concat(int64, fletcher_chunked[int64] yields int64")
        elif pa.types.is_temporal(arrow_dtype):
            # https://github.com/pandas-dev/pandas/issues/33331
            pytest.xfail("pd.concat(temporal, categorical) mangles dates")
        else:
            BaseReshapingTests.test_concat_mixed_dtypes(self, data)

    def test_merge_on_extension_array(self, data):
        if pa.types.is_list(data.dtype.arrow_dtype):
            pytest.xfail("pandas tries to hash scalar lists")
        else:
            BaseReshapingTests.test_merge_on_extension_array(self, data)

    def test_merge_on_extension_array_duplicates(self, data):
        if pa.types.is_list(data.dtype.arrow_dtype):
            pytest.xfail("pandas tries to hash scalar lists")
        else:
            BaseReshapingTests.test_merge_on_extension_array_duplicates(self, data)

    @xfail_list_setitem_not_implemented
    def test_ravel(self, data):
        BaseReshapingTests.test_ravel(self, data)

    @xfail_list_setitem_not_implemented
    @pytest.mark.xfail(reason="Views don't update their parent #96")
    def test_transpose(self, data):
        if hasattr(BaseReshapingTests, "test_transpose"):
            BaseReshapingTests.test_transpose(self, data)


class TestBaseSetitemTests(BaseSetitemTests):
    @xfail_list_setitem_not_implemented
    def test_setitem_scalar_series(self, data, box_in_series):
        BaseSetitemTests.test_setitem_scalar_series(self, data, box_in_series)

    @xfail_list_setitem_not_implemented
    def test_setitem_sequence(self, data, box_in_series):
        BaseSetitemTests.test_setitem_sequence(self, data, box_in_series)

    @xfail_list_setitem_not_implemented
    def test_setitem_empty_indxer(self, data, box_in_series):
        if hasattr(BaseSetitemTests, "test_setitem_empty_indxer"):
            BaseSetitemTests.test_setitem_empty_indxer(self, data, box_in_series)

    @xfail_list_setitem_not_implemented
    def test_setitem_empty_indexer(self, data, box_in_series):
        if hasattr(BaseSetitemTests, "test_setitem_empty_indexer"):
            BaseSetitemTests.test_setitem_empty_indexer(self, data, box_in_series)

    @xfail_list_setitem_not_implemented
    def test_setitem_sequence_broadcasts(self, data, box_in_series):
        BaseSetitemTests.test_setitem_sequence_broadcasts(self, data, box_in_series)

    @pytest.mark.parametrize("setter", ["loc", "iloc"])
    @xfail_list_setitem_not_implemented
    def test_setitem_scalar(self, data, setter):
        BaseSetitemTests.test_setitem_scalar(self, data, setter)

    @xfail_list_setitem_not_implemented
    def test_setitem_loc_scalar_mixed(self, data):
        BaseSetitemTests.test_setitem_loc_scalar_mixed(self, data)

    @xfail_list_setitem_not_implemented
    def test_setitem_loc_scalar_single(self, data):
        BaseSetitemTests.test_setitem_loc_scalar_single(self, data)

    @xfail_list_setitem_not_implemented
    def test_setitem_loc_scalar_multiple_homogoneous(self, data):
        BaseSetitemTests.test_setitem_loc_scalar_multiple_homogoneous(self, data)

    @xfail_list_setitem_not_implemented
    def test_setitem_iloc_scalar_mixed(self, data):
        BaseSetitemTests.test_setitem_iloc_scalar_mixed(self, data)

    @xfail_list_setitem_not_implemented
    def test_setitem_iloc_scalar_single(self, data):
        BaseSetitemTests.test_setitem_iloc_scalar_single(self, data)

    @xfail_list_setitem_not_implemented
    def test_setitem_iloc_scalar_multiple_homogoneous(self, data):
        BaseSetitemTests.test_setitem_iloc_scalar_multiple_homogoneous(self, data)

    @xfail_list_setitem_not_implemented
    def test_setitem_nullable_mask(self, data):
        if not PANDAS_GE_1_1_0:
            BaseSetitemTests.test_setitem_nullable_mask(self, data)

    @pytest.mark.parametrize("as_callable", [True, False])
    @pytest.mark.parametrize("setter", ["loc", None])
    @xfail_list_setitem_not_implemented
    def test_setitem_mask_aligned(self, data, as_callable, setter):
        BaseSetitemTests.test_setitem_mask_aligned(self, data, as_callable, setter)

    @pytest.mark.parametrize("setter", ["loc", None])
    @xfail_list_setitem_not_implemented
    def test_setitem_mask_broadcast(self, data, setter):
        BaseSetitemTests.test_setitem_mask_broadcast(self, data, setter)

    @xfail_list_setitem_not_implemented
    def test_setitem_slice(self, data, box_in_series):
        if PANDAS_GE_1_1_0:
            BaseSetitemTests.test_setitem_slice(self, data, box_in_series)

    @xfail_list_setitem_not_implemented
    def test_setitem_loc_iloc_slice(self, data):
        if PANDAS_GE_1_1_0:
            BaseSetitemTests.test_setitem_loc_iloc_slice(self, data)

    @xfail_list_setitem_not_implemented
    def test_setitem_slice_array(self, data):
        BaseSetitemTests.test_setitem_slice_array(self, data)

    @pytest.mark.xfail(reason="GH#20441: setitem on extension types.")
    @xfail_list_setitem_not_implemented
    def test_setitem_tuple_index(self, data):
        if hasattr(BaseSetitemTests, "test_setitem_tuple_index"):
            BaseSetitemTests.test_setitem_tuple_index(self, data)

    @xfail_list_setitem_not_implemented
    @pytest.mark.parametrize(
        "mask",
        [
            np.array([True, True, True, False, False]),
            pd.array([True, True, True, False, False], dtype="boolean"),
            pd.array([True, True, True, pd.NA, pd.NA], dtype="boolean"),
        ],
        ids=["numpy-array", "boolean-array", "boolean-array-na"],
    )
    def test_setitem_mask(self, data, mask, box_in_series):
        if PANDAS_GE_1_1_0:
            BaseSetitemTests.test_setitem_mask(self, data, mask, box_in_series)

    @pytest.mark.xfail(reason="Views don't update their parent #96")
    def test_setitem_preserves_views(self, data):
        pass

    @xfail_list_setitem_not_implemented
    def test_setitem_mask_boolean_array_with_na(self, data, box_in_series):
        if PANDAS_GE_1_1_0:
            BaseSetitemTests.test_setitem_mask_boolean_array_with_na(
                self, data, box_in_series
            )

    @pytest.mark.parametrize(
        "idx",
        [[0, 1, 2], pd.array([0, 1, 2], dtype="Int64"), np.array([0, 1, 2])],
        ids=["list", "integer-array", "numpy-array"],
    )
    @pytest.mark.xfail(reason="https://github.com/xhochy/fletcher/issues/110")
    def test_setitem_integer_array(self, data, idx, box_in_series):
        if PANDAS_GE_1_1_0:
            BaseSetitemTests.test_setitem_integer_array(self, data, idx, box_in_series)


class TestBaseParsingTests(BaseParsingTests):
    @pytest.mark.parametrize("engine", ["c", "python"])
    def test_EA_types(self, engine, data):
        pytest.mark.xfail(
            "pandas doesn't yet support registering ExtentionDtypes via a pattern"
        )


class TestBasePrintingTests(BasePrintingTests):
    pass


class TestBaseBooleanReduceTests(BaseBooleanReduceTests):
    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series(self, data, all_boolean_reductions, skipna):
        if pa.types.is_boolean(data.dtype.arrow_dtype):
            BaseBooleanReduceTests.test_reduce_series(
                self, data, all_boolean_reductions, skipna
            )
        else:
            pytest.skip("Boolean reductions are only tested with boolean types")


class TestBaseNoReduceTests(BaseNoReduceTests):
    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series_numeric(self, data, all_numeric_reductions, skipna):
        arrow_dtype = data.dtype.arrow_dtype
        if (
            pa.types.is_integer(arrow_dtype)
            or pa.types.is_floating(arrow_dtype)
            or pa.types.is_decimal(arrow_dtype)
        ):
            pytest.skip("Numeric arrays implement reductions, so don't raise")
        else:
            BaseNoReduceTests.test_reduce_series_numeric(
                self, data, all_numeric_reductions, skipna
            )

    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series_boolean(self, data, all_boolean_reductions, skipna):
        if pa.types.is_boolean(data.dtype.arrow_dtype):
            pytest.skip("BooleanArray does define boolean reductions, so don't raise")
        else:
            BaseNoReduceTests.test_reduce_series_boolean(
                self, data, all_boolean_reductions, skipna
            )


class TestBaseNumericReduceTests(BaseNumericReduceTests):
    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series(self, data, all_numeric_reductions, skipna):
        if all_numeric_reductions == "prod":
            # Shorten in the case of prod to avoid overflows
            data = data[:2]
        arrow_dtype = data.dtype.arrow_dtype
        if (
            pa.types.is_integer(arrow_dtype)
            or pa.types.is_floating(arrow_dtype)
            or pa.types.is_decimal(arrow_dtype)
        ):
            BaseNumericReduceTests.test_reduce_series(
                self, data, all_numeric_reductions, skipna
            )
        else:
            pytest.skip("Reduce not implemented on non-numeric types")


class TestBaseComparisonOpsTests(BaseComparisonOpsTests):
    def check_opname(self, s, op_name, other, exc=None):
        super().check_opname(s, op_name, other, exc=None)

    def _compare_other(self, s, data, op_name, other):
        self.check_opname(s, op_name, other)

    def test_compare_scalar(self, data, all_compare_operators):
        if pa.types.is_list(data.dtype.arrow_dtype):
            pytest.xfail("pandas cannot cope with lists as scalar")
        else:
            # FIXME: Upstream always compares againt 0
            op_name = all_compare_operators
            s = pd.Series(data)
            self._compare_other(s, data, op_name, data[0])

    def test_compare_array(self, data, all_compare_operators):
        if pa.types.is_list(data.dtype.arrow_dtype):
            pytest.xfail("Comparision of list not implemented yet")
        else:
            BaseComparisonOpsTests.test_compare_array(self, data, all_compare_operators)

    def _check_op(self, s, op, other, op_name, exc=NotImplementedError):
        if exc is None:
            result = op(s, other)
            # We return fletcher booleans to support nulls
            expected = s.combine(other, op)
            if not isinstance(expected.dtype, FletcherBaseDtype):
                expected = pd.Series(type(s.values)(expected.values))
            self.assert_series_equal(result, expected)
        else:
            with pytest.raises(exc):
                op(s, other)


class TestBaseArithmeticOpsTests(BaseArithmeticOpsTests):
    # TODO: Instead of skipping other types, we should set the correct exceptions here
    series_scalar_exc: Optional[Type[TypeError]] = None
    frame_scalar_exc: Optional[Type[TypeError]] = None
    series_array_exc: Optional[Type[TypeError]] = None
    divmod_exc: Optional[Type[TypeError]] = None

    def _check_op(self, s, op, other, op_name, exc=NotImplementedError):
        if exc is None:
            result = op(s, other)
            if hasattr(self, "_combine"):
                expected = self._combine(s, other, op)
            else:
                if isinstance(s, pd.DataFrame):
                    if len(s.columns) != 1:
                        raise NotImplementedError()
                    expected = s.iloc[:, 0].combine(other, op).to_frame()
                else:
                    expected = s.combine(other, op)
            expected_dtype = get_dtype(expected)
            s_dtype = get_dtype(s)

            # Combine always returns an int64 for integral arrays but for
            # operations on smaller integer types, we expect also smaller int types
            # in the result of the non-combine operations.
            if hasattr(expected_dtype, "arrow_dtype"):
                arrow_dtype = expected_dtype.arrow_dtype
                if pa.types.is_integer(arrow_dtype):
                    # In the case of an operand with a higher bytesize, we also expect the
                    # output to be int64.
                    other_is_np_int64 = (
                        isinstance(other, pd.Series)
                        and isinstance(other.values, np.ndarray)
                        and other.dtype.char in ("q", "l")
                    )
                    if (
                        pa.types.is_integer(arrow_dtype)
                        and pa.types.is_integer(s_dtype.arrow_dtype)
                        and not other_is_np_int64
                    ):
                        expected = expected.astype(s_dtype)

            self.assert_equal(result, expected)
        else:
            with pytest.raises(exc):
                op(s, other)

    @skip_non_artithmetic_type
    def test_arith_series_with_scalar(self, data, all_arithmetic_operators):
        BaseArithmeticOpsTests.test_arith_series_with_scalar(
            self, data, all_arithmetic_operators
        )

    @skip_non_artithmetic_type
    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        BaseArithmeticOpsTests.test_arith_series_with_array(
            self, data, all_arithmetic_operators
        )

    @skip_non_artithmetic_type
    def test_divmod(self, data):
        BaseArithmeticOpsTests.test_divmod(self, data)

    @skip_non_artithmetic_type
    def test_divmod_series_array(self, data, data_for_twos):
        BaseArithmeticOpsTests.test_divmod(self, data)

    @skip_non_artithmetic_type
    def test_add_series_with_extension_array(self, data):
        BaseArithmeticOpsTests.test_add_series_with_extension_array(self, data)

    @skip_non_artithmetic_type
    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators):
        if hasattr(BaseArithmeticOpsTests, "test_arith_frame_with_scalar"):
            BaseArithmeticOpsTests.test_arith_frame_with_scalar(
                self, data, all_arithmetic_operators
            )

    def test_error(self, data, all_arithmetic_operators):
        arrow_dtype = data.dtype.arrow_dtype
        if (
            pa.types.is_integer(arrow_dtype)
            or pa.types.is_floating(arrow_dtype)
            or pa.types.is_decimal(arrow_dtype)
        ):
            pytest.skip("Numeric does not error on ops")
        else:
            pytest.xfail("Should error here")

    def _check_divmod_op(self, s, op, other, exc=None):
        super()._check_divmod_op(s, op, other, None)
