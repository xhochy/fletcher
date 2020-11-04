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
    return fr.FletcherChunkedArray(chunked_array)


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
        fr.FletcherChunkedArray(None)


def test_flatten():
    list_array = pa.array([[1, 2], [3, 4]])
    npt.assert_array_equal(
        fr.FletcherContinuousArray(list_array).flatten(), [1, 2, 3, 4]
    )

    chunked_list_array = pa.chunked_array([list_array, list_array])
    npt.assert_array_equal(
        fr.FletcherChunkedArray(chunked_list_array).flatten(), [1, 2, 3, 4, 1, 2, 3, 4]
    )


def test_pandas_from_arrow():
    arr = pa.array(["a", "b", "c"], pa.string())

    expected_series_woutname = pd.Series(fr.FletcherChunkedArray(arr))
    pdt.assert_series_equal(expected_series_woutname, fr.pandas_from_arrow(arr))

    expected_series_woutname = pd.Series(fr.FletcherContinuousArray(arr))
    pdt.assert_series_equal(
        expected_series_woutname, fr.pandas_from_arrow(arr, continuous=True)
    )

    rb = pa.RecordBatch.from_arrays([arr], ["column"])
    expected_df = pd.DataFrame({"column": fr.FletcherChunkedArray(arr)})
    table = pa.Table.from_arrays([arr], ["column"])
    pdt.assert_frame_equal(expected_df, fr.pandas_from_arrow(rb))
    pdt.assert_frame_equal(expected_df, fr.pandas_from_arrow(table))

    expected_df = pd.DataFrame({"column": fr.FletcherContinuousArray(arr)})
    table = pa.Table.from_arrays([arr], ["column"])
    pdt.assert_frame_equal(expected_df, fr.pandas_from_arrow(rb, continuous=True))
    pdt.assert_frame_equal(expected_df, fr.pandas_from_arrow(table, continuous=True))


def test_take_list_arrays():
    indices = [0, 1, 4, 3, 5]
    indptr = [0, 2, 3, 5]
    list_array = pa.ListArray.from_arrays(indptr, indices)
    large_list_array = pa.LargeListArray.from_arrays(indptr, indices)

    test_with_null = fr.FletcherContinuousArray(pa.array([[1, 2], [None, 3], [4, 5]]))

    assert np.all(
        pa.array(test_with_null.take([1, 2, 1])).to_pylist()
        == [[None, 3], [4, 5], [None, 3]]
    )

    test = fr.FletcherContinuousArray(pa.chunked_array([list_array, list_array])).take(
        [0, 5, 1]
    )
    test_large = fr.FletcherContinuousArray(
        pa.chunked_array([large_list_array, large_list_array])
    ).take([0, 5, 1])
    expected = [[0, 1], [3, 5], [4]]
    assert np.all(
        list(
            map(
                lambda x: np.all(
                    np.array(test[x]) == np.array(expected, dtype="object")[x]
                ),
                range(0, len(test)),
            )
        )
    )
    assert np.all(
        list(
            map(
                lambda x: np.all(
                    np.array(test_large[x]) == np.array(expected, dtype="object")[x]
                ),
                range(0, len(test_large)),
            )
        )
    )
