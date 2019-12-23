import numpy as np
import pandas as pd
import pyarrow as pa

import fletcher as fr


class ComparisonOps:
    def setup(self):
        data_a = np.random.randint(0, 2 ** 20, size=2 ** 24)
        data_b = np.random.randint(0, 2 ** 20, size=2 ** 24)
        self.pd_int_a = pd.Series(data_a)
        self.pd_int_b = pd.Series(data_b)
        self.fr_cont_int_a = pd.Series(fr.FletcherContinuousArray(data_a))
        self.fr_cont_int_b = pd.Series(fr.FletcherContinuousArray(data_b))
        self.fr_chunked_int_a = pd.Series(fr.FletcherChunkedArray(data_a))
        self.fr_chunked_int_b = pd.Series(fr.FletcherChunkedArray(data_b))

        mask_a = np.random.rand(2 ** 24) > 0.8
        mask_b = np.random.rand(2 ** 24) > 0.8
        self.pd_int_na_a = pd.Series(pd.arrays.IntegerArray(data_a, mask_a))
        self.pd_int_na_b = pd.Series(pd.arrays.IntegerArray(data_b, mask_b))
        self.fr_cont_int_na_a = pd.Series(
            fr.FletcherContinuousArray(pa.array(data_a, mask=mask_a))
        )
        self.fr_cont_int_na_b = pd.Series(
            fr.FletcherContinuousArray(pa.array(data_b, mask=mask_b))
        )
        self.fr_chunked_int_na_a = pd.Series(
            fr.FletcherChunkedArray(pa.array(data_a, mask=mask_a))
        )
        self.fr_chunked_int_na_b = pd.Series(
            fr.FletcherChunkedArray(pa.array(data_b, mask=mask_b))
        )

    def time_pd_lt(self):
        self.pd_int_a < self.pd_int_b

    def time_fr_cont_lt(self):
        self.fr_cont_int_a < self.fr_cont_int_b

    def time_fr_chunked_lt(self):
        self.fr_chunked_int_a < self.fr_chunked_int_b

    def time_pd_lt_na(self):
        self.pd_int_na_a < self.pd_int_na_b

    def time_fr_cont_lt_na(self):
        self.fr_cont_int_na_a < self.fr_cont_int_na_b

    def time_fr_chunked_lt_na(self):
        self.fr_chunked_int_na_a < self.fr_chunked_int_na_b
