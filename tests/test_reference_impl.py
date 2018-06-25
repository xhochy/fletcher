# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import operator as op

import pandas as pd
import pandas.util.testing as pdt
import pyarrow as pa
import pytest

import fletcher as fr


data = ["foo", None, "baz", "bar", None, "..bar"]

df = pd.DataFrame(
    {"pd": pd.Series(data), "fr": fr.FletcherArray(data, dtype=pa.string())}
)


# syntactic sugar to make test cases easier to read
class Case:

    def __init__(self, label):
        self._label = label

    def __getattr__(self, name):
        return lambda *args, **kwargs: dict(
            label=self._label, method=name, args=args, kwargs=kwargs
        )


test_cases = [
    Case("startswith").startswith("ba"),
    Case("endswith").endswith("ar"),
    Case("startswith with na").startswith("ba", na=False),
    Case("endswith with na").endswith("ar", na=False),
]


@pytest.mark.parametrize("spec", test_cases, ids=op.itemgetter("label"))
def test_reference_impl(spec):
    expected = getattr(df["pd"].str, spec["method"])(
        *spec.get("args", []), **spec.get("kwargs", {})
    )
    actual = getattr(df["fr"].text, spec["method"])(
        *spec.get("args", []), **spec.get("kwargs", {})
    )

    pdt.assert_series_equal(expected, actual, check_names=False)
