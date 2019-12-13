import numpy as np
import pandas as pd
import pyarrow as pa

import fletcher as fr


class ArithmeticOps:
    def setup(self):
        data = np.random.randint(0, 2 ** 20, size=2 ** 24)
        self.pd_int = pd.Series(data)
        self.fr_cont_int = pd.Series(fr.FletcherContinuousArray(data))
        self.fr_chunked_int = pd.Series(fr.FletcherChunkedArray(data))

        mask = np.random.rand(2 ** 24) > 0.8
        self.pd_int_na = pd.Series(pd.arrays.IntegerArray(data, mask))
        self.fr_cont_int_na = pd.Series(
            fr.FletcherContinuousArray(pa.array(data, mask=mask))
        )
        self.fr_chunked_int_na = pd.Series(
            fr.FletcherChunkedArray(pa.array(data, mask=mask))
        )

    def time_pd_sum(self):
        self.pd_int.sum()

    def time_fr_cont_sum(self):
        self.fr_cont_int.sum()

    def time_fr_chunked_sum(self):
        self.fr_chunked_int.sum()

    def time_pd_sum_na(self):
        self.pd_int_na.sum()

    def time_fr_cont_sum_na(self):
        self.fr_cont_int_na.sum()

    def time_fr_chunked_sum_na(self):
        self.fr_chunked_int_na.sum()
