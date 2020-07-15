import pandas as pd
import pytest

import fletcher as fr


@pytest.mark.parametrize(
    ["data", "slice_", "expected"],
    [
        (["abcd", "defg", "hijk"], (1, 2, 1), ["b", "e", "i"]),
        (["abcd", "defg", "h"], (1, 2, 1), ["b", "e", ""]),
        (["abcd", "defg", "hijk"], (1, 4, 2), ["bd", "eg", "ik"]),
        (["abcd", "defg", "hijk"], (0, -2, 1), ["ab", "de", "hi"]),
        (["abcd", "defg", "hijk"], (-5, -2, 1), ["ab", "de", "hi"]),
        (["aécd", "d🙂fg", "écijk"], (1, 2, 1), ["é", "🙂", "c"]),
        (["abcd", "defg", "hijk"], (3, 1, -1), ["dc", "gf", "kj"]),
        (["abcd", "defg", "hijk"], (5, 0, -2), ["db", "ge", "ki"]),
        (["abcd", "defg", "hijk"], (3, 20, 1), ["d", "g", "k"]),
        (["abcd", "defg", "hijk"], (10, 20, 1), ["", "", ""]),
        (["abcd", "defg", None], (10, 20, 1), ["", "", None]),
        (["abcd", "defg", "hijk"], (1, None, 1), ["bcd", "efg", "ijk"]),
    ],
)
@pytest.mark.parametrize(
    "storage_type", [fr.FletcherContinuousArray, fr.FletcherChunkedArray]
)
def test_slice(data, slice_, expected, storage_type):
    fr_series = pd.Series(storage_type(data))
    fr_out = fr_series.fr_str.slice(*slice_).astype(object)
    pd.testing.assert_series_equal(fr_out, pd.Series(expected))

    pd_out = pd.Series(data).str.slice(*slice_)
    pd.testing.assert_series_equal(fr_out, pd_out)
