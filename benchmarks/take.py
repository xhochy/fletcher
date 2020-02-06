import itertools
from functools import partialmethod

import numpy as np
import pandas as pd
import pyarrow as pa

import fletcher as fr


def _take_nofill_range(self, attr_name: str) -> None:
    attr = getattr(self, attr_name)
    attr.take(np.arange(len(attr) / 2))


def _take_nofill_random(self, attr_name: str) -> None:
    attr = getattr(self, attr_name)
    attr.take(self.data_small)


def _take_nofill_random_negative(self, attr_name: str) -> None:
    attr = getattr(self, attr_name)
    attr.take(-self.data_small)


# TODO: How is this trigged? pd.Series.take doesn't have a fill_value parameter
def _take_fill_random(self, attr_name: str) -> None:
    attr = getattr(self, attr_name)
    attr.take(self.data_small_missing, allow_fill=True, fill_value=attr[0])


class Take:
    def setup(self):
        np.random.seed(93487)
        # TODO: Is it maybe faster to separate each type into its own Take* class?
        #       It seems like the data is regenerated for each benchmark and thus
        #       is quite some overhead here.
        self.data = np.random.randint(0, 2 ** 20, size=2 ** 24)
        self.pd_int = pd.Series(self.data)
        self.fr_cont_int = pd.Series(fr.FletcherContinuousArray(self.data))
        chunked_data = pa.chunked_array(
            [
                pa.array(self.data[0 : len(self.data) // 2]),
                pa.array(self.data[len(self.data) // 2 : -1]),
            ]
        )
        self.fr_chunked_int = pd.Series(fr.FletcherChunkedArray(chunked_data))

        mask = np.random.rand(2 ** 24) > 0.8
        self.pd_int_na = pd.Series(pd.arrays.IntegerArray(self.data, mask))
        self.fr_cont_int_na = pd.Series(
            fr.FletcherContinuousArray(pa.array(self.data, mask=mask))
        )
        self.fr_chunked_int_na = pd.Series(
            fr.FletcherChunkedArray(pa.array(self.data, mask=mask))
        )

        self.data_small = np.random.randint(0, 2 ** 16, size=2 ** 18)
        self.data_small_missing = self.data_small.copy()
        self.data_small_missing[0:-1:2] = -1
        data_small_str = self.data_small.astype(str)
        self.pd_str = pd.Series(data_small_str)
        self.fr_cont_str = pd.Series(fr.FletcherContinuousArray(data_small_str))
        data_small_str_chunked = pa.chunked_array(
            [
                pa.array(data_small_str[0 : len(data_small_str) // 2]),
                pa.array(data_small_str[len(data_small_str) // 2 : -1]),
            ]
        )
        self.fr_chunked_str = pd.Series(fr.FletcherChunkedArray(data_small_str_chunked))


# Auto-generate benchmark methods
attrs = [
    "pd_int",
    "fr_cont_int",
    "fr_chunked_int",
    "pd_int_na",
    "fr_cont_int_na",
    "fr_chunked_int_na",
    "pd_str",
    "fr_cont_str",
    "fr_chunked_str",
]
functions = {
    "take_nofill_range": _take_nofill_range,
    "take_nofill_random": _take_nofill_random,
    "take_nofill_random_negative": _take_nofill_random_negative,
    # "take_fill_random": _take_fill_random,
}

for attr, func in itertools.product(attrs, functions.items()):
    fname, call = func
    setattr(Take, f"time_{fname}_{attr}", partialmethod(call, attr_name=attr))
