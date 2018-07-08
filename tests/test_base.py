# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

from distutils.version import LooseVersion

import numpy as np
import numpy.testing as npt
import pandas as pd
import pyarrow as pa
import pytest

import fletcher as fl


@pytest.fixture
def array_inhom_chunks():
    chunk1 = pa.array(list("abc"), pa.string())
    chunk2 = pa.array(list("12345"), pa.string())
    chunk3 = pa.array(list("Z"), pa.string())
    chunked_array = pa.chunked_array([chunk1, chunk2, chunk3])
    return fl.FletcherArray(chunked_array)


def test_get_chunk_offsets(array_inhom_chunks):
    actual = array_inhom_chunks._get_chunk_offsets()
    expected = np.array([0, 3, 8])
    npt.assert_array_equal(actual, expected)


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


@pytest.mark.skipif(
    LooseVersion(pd.__version__) < "0.24.0dev0", reason="Pandas version is too old."
)
def test_from_pandas_registry():
    ser = pd.Series([1, 2, 3], dtype="fletcher[int64]")
    assert ser.dtype == fl.FletcherDtype("int64")
    assert isinstance(ser.values, fl.FletcherArray)

    ser = pd.Series([1, 2, 3], dtype="Fletcher[int64]")
    assert ser.dtype == fl.FletcherDtype("int64")
    assert isinstance(ser.values, fl.FletcherArray)
