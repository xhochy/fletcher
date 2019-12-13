import datetime
import string
import sys
from collections import namedtuple
from distutils.version import LooseVersion
from typing import Optional, Type

import pandas as pd
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

from fletcher import (
    FletcherBaseDtype,
    FletcherChunkedArray,
    FletcherChunkedDtype,
    FletcherContinuousArray,
    FletcherContinuousDtype,
)

if LooseVersion(pd.__version__) >= "0.25.0":
    # imports of pytest fixtures needed for derived unittest classes
    from pandas.tests.extension.conftest import (  # noqa: F401
        as_array,  # noqa: F401
        use_numpy,  # noqa: F401
        groupby_apply_op,  # noqa: F401
        as_frame,  # noqa: F401
        as_series,  # noqa: F401
    )


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


@pytest.fixture(params=["chunked", "continuous"])
def fletcher_variant(request):
    """Whether to test the chunked or continuous implementation."""
    return request.param


@pytest.fixture
def fletcher_dtype(fletcher_variant):
    if fletcher_variant == "chunked":
        return FletcherChunkedDtype
    else:
        return FletcherContinuousDtype


@pytest.fixture
def fletcher_array(fletcher_variant):
    if fletcher_variant == "chunked":
        return FletcherChunkedArray
    else:
        return FletcherContinuousArray


@pytest.fixture(params=[True, False])
def box_in_series(request):
    """Whether to box the data in a Series."""
    return request.param


@pytest.fixture(params=test_types)
def fletcher_type(request):
    return request.param


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


class TestBaseCasting(BaseCastingTests):
    pass


class TestBaseConstructors(BaseConstructorsTests):
    pass


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

    def test_take_series(self, data):
        BaseGetitemTests.test_take_series(self, data)

    def test_loc_iloc_frame_single_dtype(self, data):
        if pa.types.is_string(data.dtype.arrow_dtype):
            pytest.mark.xfail(
                reason="https://github.com/pandas-dev/pandas/issues/27673"
            )
        else:
            BaseGetitemTests.test_loc_iloc_frame_single_dtype(self, data)

    @pytest.mark.skip
    def test_reindex(self):
        # No longer available in master and fails with pandas 0.23.1
        # due to a dtype assumption that does not hold for Arrow
        pass


class TestBaseGroupbyTests(BaseGroupbyTests):
    @pytest.mark.parametrize("as_index", [True, False])
    def test_groupby_extension_agg(self, as_index, data_for_grouping):
        if (
            pa.types.is_integer(data_for_grouping.dtype.arrow_dtype)
            or pa.types.is_floating(data_for_grouping.dtype.arrow_dtype)
            or pa.types.is_boolean(data_for_grouping.dtype.arrow_dtype)
        ):
            pytest.mark.xfail(reason="ExtensionIndex is not yet implemented")
        elif pa.types.is_list(data_for_grouping.dtype.arrow_dtype):
            pytest.mark.xfail(
                reason="ArrowNotImplementedError: dictionary-encode not implemented for list<item: string>"
            )
        else:
            BaseGroupbyTests.test_groupby_extension_agg(
                self, as_index, data_for_grouping
            )

    def test_groupby_extension_no_sort(self, data_for_grouping):
        if (
            pa.types.is_integer(data_for_grouping.dtype.arrow_dtype)
            or pa.types.is_floating(data_for_grouping.dtype.arrow_dtype)
            or pa.types.is_boolean(data_for_grouping.dtype.arrow_dtype)
        ):
            pytest.mark.xfail(reasion="ExtensionIndex is not yet implemented")
        elif pa.types.is_list(data_for_grouping.dtype.arrow_dtype):
            pytest.mark.xfail(
                reason="ArrowNotImplementedError: dictionary-encode not implemented for list<item: string>"
            )
        else:
            BaseGroupbyTests.test_groupby_extension_no_sort(self, data_for_grouping)

    def test_groupby_extension_transform(self, data_for_grouping):
        if pa.types.is_boolean(data_for_grouping.dtype.arrow_dtype):
            valid = data_for_grouping[~data_for_grouping.isna()]
            df = pd.DataFrame({"A": [1, 1, 3, 3, 1, 4], "B": valid})

            result = df.groupby("B").A.transform(len)
            # Expected grouping is different as we only have two non-null values
            expected = pd.Series([3, 3, 3, 3, 3, 3], name="A")

            self.assert_series_equal(result, expected)
        elif pa.types.is_list(data_for_grouping.dtype.arrow_dtype):
            pytest.mark.xfail(
                reason="ArrowNotImplementedError: dictionary-encode not implemented for list<item: string>"
            )
        else:
            BaseGroupbyTests.test_groupby_extension_transform(self, data_for_grouping)

    def test_groupby_extension_apply(
        self, data_for_grouping, groupby_apply_op  # noqa: F811
    ):
        if pa.types.is_list(data_for_grouping.dtype.arrow_dtype):
            pytest.mark.xfail(
                reason="ArrowNotImplementedError: dictionary-encode not implemented for list<item: string>"
            )
        else:
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

    def test_copy(self, data):
        if pa.types.is_list(data.dtype.arrow_dtype):
            pytest.xfail("Scalar __setitem__ not implemented for list")
        else:
            BaseInterfaceTests.test_array_interface(self, data)


class TestBaseMethodsTests(BaseMethodsTests):

    # https://github.com/pandas-dev/pandas/issues/22843
    @pytest.mark.skip(reason="Incorrect expected")
    @pytest.mark.parametrize("dropna", [True, False])
    def test_value_counts(self, all_data, dropna, dtype):
        pass

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

    # @pytest.mark.parametrize("na_sentinel", [-1, -2])
    # def test_factorize(self, data_for_grouping, na_sentinel):
    #    BaseMethodsTests.test_factorize(self, data_for_grouping, na_sentinel)

    def test_argsort(self, data_for_sorting):
        if pa.types.is_boolean(data_for_sorting.dtype.arrow_dtype):
            pytest.skip("Boolean has too few values for this test")
        else:
            BaseMethodsTests.test_argsort(self, data_for_sorting)

    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values(self, data_for_sorting, ascending):
        if pa.types.is_boolean(data_for_sorting.dtype.arrow_dtype):
            pytest.skip("Boolean has too few values for this test")
        else:
            BaseMethodsTests.test_sort_values(self, data_for_sorting, ascending)

    @pytest.mark.parametrize("na_sentinel", [-1, -2])
    def test_factorize(self, data_for_grouping, na_sentinel):
        if pa.types.is_boolean(data_for_grouping.dtype.arrow_dtype):
            pytest.skip("Boolean has too few values for this test")
        elif pa.types.is_list(data_for_grouping.dtype.arrow_dtype):
            pytest.xfail("Factorize not iplemented for lists")
        else:
            BaseMethodsTests.test_factorize(self, data_for_grouping, na_sentinel)

    @pytest.mark.parametrize("na_sentinel", [-1, -2])
    def test_factorize_equivalence(self, data_for_grouping, na_sentinel):
        if pa.types.is_boolean(data_for_grouping.dtype.arrow_dtype):
            pytest.skip("Boolean has too few values for this test")
        elif pa.types.is_list(data_for_grouping.dtype.arrow_dtype):
            pytest.xfail("Factorize not iplemented for lists")
        else:
            BaseMethodsTests.test_factorize_equivalence(
                self, data_for_grouping, na_sentinel
            )

    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values_frame(self, data_for_sorting, ascending):
        if pa.types.is_list(data_for_sorting.dtype.arrow_dtype):
            pytest.xfail("Factorize not iplemented for lists")
        else:
            BaseMethodsTests.test_sort_values_frame(self, data_for_sorting, ascending)

    def test_searchsorted(self, data_for_sorting, as_series):  # noqa: F811
        if pa.types.is_boolean(data_for_sorting.dtype.arrow_dtype):
            pytest.skip("Boolean has too few values for this test")
        else:
            BaseMethodsTests.test_searchsorted(self, data_for_sorting, as_series)

    @pytest.mark.parametrize("box", [pd.Series, lambda x: x])
    @pytest.mark.parametrize("method", [lambda x: x.unique(), pd.unique])
    def test_unique(self, data, box, method):
        if pa.types.is_list(data.dtype.arrow_dtype):
            pytest.xfail("unique is not implemented for lists")
        else:
            BaseMethodsTests.test_unique(self, data, box, method)

    def test_factorize_empty(self, data):
        if pa.types.is_list(data.dtype.arrow_dtype):
            pytest.xfail("dictionary-encode is not implemented for lists")
        else:
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

    def test_combine_first(self, data):
        if pa.types.is_list(data.dtype.arrow_dtype):
            pytest.xfail("Scalar __setitem__ not implemented for list")
        else:
            BaseMethodsTests.test_combine_first(self, data)

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

    def test_where_series(self, data, na_value, as_frame):  # noqa: F811
        if pa.types.is_list(data.dtype.arrow_dtype):
            pytest.xfail("Scalar __setitem__ not implemented for list")
        else:
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
        if dtype.name in [
            "fletcher_chunked[int64]",
            "fletcher_chunked[double]",
            "fletcher_chunked[bool]",
            "fletcher_continuous[int64]",
            "fletcher_continuous[double]",
            "fletcher_continuous[bool]",
        ]:
            # https://github.com/pandas-dev/pandas/issues/21792
            pytest.skip("pd.concat(int64, fletcher_chunked[int64] yields int64")
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

    def test_ravel(self, data):
        if pa.types.is_list(data.dtype.arrow_dtype):
            pytest.xfail("Scalar __setitem__ not implemented for list")
        else:
            BaseReshapingTests.test_ravel(self, data)


class TestBaseSetitemTests(BaseSetitemTests):
    def test_setitem_scalar_series(self, data, box_in_series):
        if pa.types.is_list(data.dtype.arrow_dtype):
            pytest.xfail("Scalar __setitem__ not implemented for list")
        else:
            BaseSetitemTests.test_setitem_scalar_series(self, data, box_in_series)

    def test_setitem_sequence(self, data, box_in_series):
        if pa.types.is_list(data.dtype.arrow_dtype):
            pytest.xfail("Scalar __setitem__ not implemented for list")
        else:
            BaseSetitemTests.test_setitem_sequence(self, data, box_in_series)

    def test_setitem_empty_indxer(self, data, box_in_series):
        if pa.types.is_list(data.dtype.arrow_dtype):
            pytest.xfail("Scalar __setitem__ not implemented for list")
        else:
            BaseSetitemTests.test_setitem_empty_indxer(self, data, box_in_series)

    def test_setitem_sequence_broadcasts(self, data, box_in_series):
        if pa.types.is_list(data.dtype.arrow_dtype):
            pytest.xfail("Scalar __setitem__ not implemented for list")
        else:
            BaseSetitemTests.test_setitem_sequence_broadcasts(self, data, box_in_series)

    @pytest.mark.parametrize("setter", ["loc", "iloc"])
    def test_setitem_scalar(self, data, setter):
        if pa.types.is_list(data.dtype.arrow_dtype):
            pytest.xfail("Scalar __setitem__ not implemented for list")
        else:
            BaseSetitemTests.test_setitem_scalar(self, data, setter)

    def test_setitem_loc_scalar_mixed(self, data):
        if pa.types.is_list(data.dtype.arrow_dtype):
            pytest.xfail("Scalar __setitem__ not implemented for list")
        else:
            BaseSetitemTests.test_setitem_loc_scalar_mixed(self, data)

    def test_setitem_loc_scalar_single(self, data):
        if pa.types.is_list(data.dtype.arrow_dtype):
            pytest.xfail("Scalar __setitem__ not implemented for list")
        else:
            BaseSetitemTests.test_setitem_loc_scalar_single(self, data)

    def test_setitem_loc_scalar_multiple_homogoneous(self, data):
        if pa.types.is_list(data.dtype.arrow_dtype):
            pytest.xfail("Scalar __setitem__ not implemented for list")
        else:
            BaseSetitemTests.test_setitem_loc_scalar_multiple_homogoneous(self, data)

    def test_setitem_iloc_scalar_mixed(self, data):
        if pa.types.is_list(data.dtype.arrow_dtype):
            pytest.xfail("Scalar __setitem__ not implemented for list")
        else:
            BaseSetitemTests.test_setitem_iloc_scalar_mixed(self, data)

    def test_setitem_iloc_scalar_single(self, data):
        if pa.types.is_list(data.dtype.arrow_dtype):
            pytest.xfail("Scalar __setitem__ not implemented for list")
        else:
            BaseSetitemTests.test_setitem_iloc_scalar_single(self, data)

    def test_setitem_iloc_scalar_multiple_homogoneous(self, data):
        if pa.types.is_list(data.dtype.arrow_dtype):
            pytest.xfail("Scalar __setitem__ not implemented for list")
        else:
            BaseSetitemTests.test_setitem_iloc_scalar_multiple_homogoneous(self, data)

    @pytest.mark.parametrize("as_callable", [True, False])
    @pytest.mark.parametrize("setter", ["loc", None])
    def test_setitem_mask_aligned(self, data, as_callable, setter):
        if pa.types.is_list(data.dtype.arrow_dtype):
            pytest.xfail("Scalar __setitem__ not implemented for list")
        else:
            BaseSetitemTests.test_setitem_mask_aligned(self, data, as_callable, setter)

    @pytest.mark.parametrize("setter", ["loc", None])
    def test_setitem_mask_broadcast(self, data, setter):
        if pa.types.is_list(data.dtype.arrow_dtype):
            pytest.xfail("Scalar __setitem__ not implemented for list")
        else:
            BaseSetitemTests.test_setitem_mask_broadcast(self, data, setter)

    def test_setitem_slice_array(self, data):
        if pa.types.is_list(data.dtype.arrow_dtype):
            pytest.xfail("Scalar __setitem__ not implemented for list")
        else:
            BaseSetitemTests.test_setitem_slice_array(self, data)


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

    def test_arith_series_with_scalar(self, data, all_arithmetic_operators):
        arrow_dtype = data.dtype.arrow_dtype
        if (
            pa.types.is_integer(arrow_dtype)
            or pa.types.is_floating(arrow_dtype)
            or pa.types.is_decimal(arrow_dtype)
        ):
            BaseArithmeticOpsTests.test_arith_series_with_scalar(
                self, data, all_arithmetic_operators
            )
        else:
            pytest.skip("Arithmetic not implemented on non-numeric types")

    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        arrow_dtype = data.dtype.arrow_dtype
        if (
            pa.types.is_integer(arrow_dtype)
            or pa.types.is_floating(arrow_dtype)
            or pa.types.is_decimal(arrow_dtype)
        ):
            BaseArithmeticOpsTests.test_arith_series_with_array(
                self, data, all_arithmetic_operators
            )
        else:
            pytest.skip("Arithmetic not implemented on non-numeric types")

    def test_divmod(self, data):
        arrow_dtype = data.dtype.arrow_dtype
        if (
            pa.types.is_integer(arrow_dtype)
            or pa.types.is_floating(arrow_dtype)
            or pa.types.is_decimal(arrow_dtype)
        ):
            BaseArithmeticOpsTests.test_divmod(self, data)
        else:
            pytest.skip("divmode not implemented on non-numeric types")

    def test_divmod_series_array(self, data, data_for_twos):
        arrow_dtype = data.dtype.arrow_dtype
        if (
            pa.types.is_integer(arrow_dtype)
            or pa.types.is_floating(arrow_dtype)
            or pa.types.is_decimal(arrow_dtype)
        ):
            BaseArithmeticOpsTests.test_divmod(self, data)
        else:
            pytest.skip("divmode not implemented on non-numeric types")

    def test_add_series_with_extension_array(self, data):
        arrow_dtype = data.dtype.arrow_dtype
        if (
            pa.types.is_integer(arrow_dtype)
            or pa.types.is_floating(arrow_dtype)
            or pa.types.is_decimal(arrow_dtype)
        ):
            BaseArithmeticOpsTests.test_add_series_with_extension_array(self, data)
        else:
            pytest.skip("not implemented on non-numeric types")

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
