import pandas as pd
import pytest

import fletcher as fr


@pytest.mark.parametrize(
    ["data", "slice_", "expected"],
    [
        (["abcd", "defg", "hijk"], (1, 2, 1), ["b", "e", "i"]),
        (
            ["abcd", "defg", "h"],
            (1, 2, 1),
            ["b", "e", ""],
        ),  # currently failing on master, works on #3
        (
            ["abcd", "defg", "hijk"],
            (1, 4, 2),
            ["bd", "eg", "ik"],
        ),  # currently failing, should be solved by #4
        (["abcd", "defg", "hijk"], (0, -2, 1), ["ab", "de", "hi"]),  # not working
        (["aécd", "d🙂fg", "écijk"], (1, 2, 1), ["é", "🙂", "c"]),
        (["abcd", "defg", "hijk"], (1, 3, -1), ["cb", "fe", "ji"]),  # not working
        (["abcd", "defg", "hijk"], (3, 20, 1), ["d", "g", "k"]),
        (["abcd", "defg", "hijk"], (10, 20, 1), ["", "", ""]),
        (["abcd", "defg", None], (10, 20, 1), ["", "", pd.np.nan]),  # not working
    ],
)
@pytest.mark.parametrize(
    "storage_type", [fr.FletcherContinuousArray, fr.FletcherChunkedArray]
)
def test_slice(data, slice_, expected, storage_type):
    """
    - test getting the right string
    - test null handling
    - test negative slice
    - test step
    - test negative step
    - test unicode
    - test outside bound -> empty string
    """
    fr_series = pd.Series(storage_type(data))
    fr_out = fr_series.text.slice(*slice_).astype(object)
    pd.testing.assert_series_equal(fr_out, pd.Series(expected))

    pd_out = pd.Series(data).str.slice(*slice_)
    pd.testing.assert_series_equal(fr_out, pd_out)