# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pyarrow as pa
import pytest

import fletcher as fr


@pytest.fixture
def array_inhom_chunks():
    chunk1 = pa.array(list("abc"), pa.string())
    chunk2 = pa.array(list("12345"), pa.string())
    chunk3 = pa.array(list("Z"), pa.string())
    chunked_array = pa.chunked_array([chunk1, chunk2, chunk3])
    return fr.FletcherArray(chunked_array)


@pytest.mark.parametrize(
    "indices, expected",
    [
        (np.array(range(3)), np.full(3, 0)),
        (np.array(range(3, 8)), np.full(5, 1)),
        (np.array([8]), np.array([2])),
        (np.array([0, 1, 5, 7, 8]), np.array([0, 0, 1, 1, 2])),
        (np.array([5, 8, 0, 7, 1]), np.array([1, 2, 0, 1, 0])),
    ],
)
def test_get_chunk_indexer(array_inhom_chunks, indices, expected):

    actual = array_inhom_chunks._get_chunk_indexer(indices)
    npt.assert_array_equal(actual, expected)


def test_fletcherarray_constructor():
    with pytest.raises(ValueError):
        fr.FletcherArray(None)


def test_pandas_from_arrow():
    arr = pa.array(["a", "b", "c"], pa.string())

    expected_series_woutname = pd.Series(fr.FletcherArray(arr))
    pdt.assert_series_equal(expected_series_woutname, fr.pandas_from_arrow(arr))

    rb = pa.RecordBatch.from_arrays([arr], ["column"])
    expected_df = pd.DataFrame({"column": fr.FletcherArray(arr)})
    pdt.assert_frame_equal(expected_df, fr.pandas_from_arrow(rb))

    table = pa.Table.from_arrays([arr], ["column"])
    pdt.assert_frame_equal(expected_df, fr.pandas_from_arrow(table))


def test_take_on_concatenated_chunks():
    test = [[1, 2, 8, 3], [4, 1, 5, 6], [7, 8, 9]]
    indices = np.array([4, 2, 8])
    expected_result = fr.FletcherArray([np.concatenate(test)[e] for e in indices])
    result = fr.FletcherArray(pa.chunked_array(test))._take_on_concatenated_chunks(
        indices
    )
    npt.assert_array_equal(expected_result, result)


def test_take_on_concatenated_chunks_with_many_chunks():
    test = [[1, 2, 3] for _ in range(100)]
    fr_test = fr.FletcherArray(pa.chunked_array(test))
    indices1 = np.array([(30 * k + (k % 3)) for k in range(0, 10)])
    indices2 = np.array([2, 5] * 100)
    for indices in [indices1, indices2]:
        expected_result = fr.FletcherArray([np.concatenate(test)[e] for e in indices])
        result = fr_test._take_on_concatenated_chunks(indices)
        npt.assert_array_equal(expected_result, result)


def test_take_on_chunks():
    test = [[1, 2, 8, 3], [4, 1, 5, 6], [7, 8, 9]]
    indices = np.array([2, 4, 8])
    limits_idx = np.array([0, 1, 2, 3])
    cum_lengths = np.array([0, 4, 8])
    expected_result = fr.FletcherArray([np.concatenate(test)[e] for e in indices])
    result = fr.FletcherArray(pa.chunked_array(test))._take_on_chunks(
        indices, limits_idx=limits_idx, cum_lengths=cum_lengths
    )
    npt.assert_array_equal(expected_result, result)


def test_take_on_chunks_with_many_chunks():
    test = [[1, 2, 3] for _ in range(100)]
    fr_test = fr.FletcherArray(pa.chunked_array(test))

    indices1 = np.array([(30 * k + (k % 3)) for k in range(0, 10)])
    # bins will be already sorted
    indices2 = np.array([2, 5] * 100)
    # bins will have to be sorted

    limits_idx1 = np.array([0] + [k // 10 for k in range(10, 110)])
    limits_idx2 = np.array([0] + [100] + [200] * 99)

    sort_idx1 = None
    sort_idx2 = np.array(
        [2 * k for k in range(0, 100)] + [2 * k + 1 for k in range(100)]
    )

    indices2 = indices2[sort_idx2]

    cum_lengths = np.array([3 * k for k in range(100)])

    for indices, limits_idx, cum_lengths, sort_idx in [
        (indices1, limits_idx1, cum_lengths, sort_idx1),
        (indices2, limits_idx2, cum_lengths, sort_idx2),
    ]:
        expected_result = fr.FletcherArray([np.concatenate(test)[e] for e in indices])
        result = fr_test._take_on_chunks(
            indices, limits_idx=limits_idx, cum_lengths=cum_lengths, sort_idx=sort_idx
        )
        npt.assert_array_equal(expected_result, result)


def test_indices_dtype():
    arr1 = fr.FletcherArray(np.zeros(np.iinfo(np.int32()).max + 1))
    arr2 = fr.FletcherArray(np.zeros(np.iinfo(np.int32()).max + 2))
    for arr in [arr1, arr2]:
        npt.assert_equal(
            len(arr) - 1, np.array([len(arr) - 1], dtype=arr._indices_dtype)[0]
        )
    npt.assert_equal(arr1._indices_dtype, np.dtype(np.int32))
    npt.assert_equal(arr2._indices_dtype, np.dtype(np.int64))


def test_take():
    test = [[1, 2, 8, 3], [4, 1, 5, 6], [7, 8, 9]]
    indices = [4, 2, 8] * 100
    fr_test = fr.FletcherArray(pa.chunked_array(test))
    result = fr_test.take(indices)
    expected_result = fr.FletcherArray(
        pa.chunked_array([[4, 8, 7] for _ in range(100)])
    )
    npt.assert_array_equal(expected_result, result)
