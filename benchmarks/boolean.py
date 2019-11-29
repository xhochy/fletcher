import numpy as np
import pandas as pd
import pyarrow as pa

import fletcher as fr


class BooleanAny:
    def setup(self):
        data = np.zeros(2 ** 24).astype(bool)
        self.fr_data = pd.Series(fr.FletcherChunkedArray(pa.array(data)))
        self.np_data = pd.Series(data.astype(np.float32))
        data_withna = np.zeros(2 ** 24).astype(bool).astype(object)
        data_withna[-1] = None
        self.fr_data_withna = pd.Series(fr.FletcherChunkedArray(pa.array(data_withna)))
        self.np_data_withna = pd.Series(data_withna.astype(np.float32))

    def time_fletcher(self):
        self.fr_data.any()

    def time_fletcher_withna(self):
        self.fr_data_withna.any()

    def track_size_fletcher(self):
        return self.fr_data.nbytes

    def track_size_fletcher_withna(self):
        return self.fr_data_withna.nbytes

    def time_numpy(self):
        self.np_data.any()

    def time_numpy_withna(self):
        self.np_data_withna.any()

    def track_size_numpy(self):
        return self.np_data.nbytes

    def track_size_numpy_withna(self):
        return self.np_data_withna.nbytes


class BooleanAll:
    def setup(self):
        data = np.ones(2 ** 24).astype(bool)
        self.fr_data = pd.Series(fr.FletcherChunkedArray(pa.array(data)))
        self.np_data = pd.Series(data.astype(np.float32))
        data_withna = data.astype(object)
        data_withna[-1] = None
        self.fr_data_withna = pd.Series(fr.FletcherChunkedArray(pa.array(data_withna)))
        self.np_data_withna = pd.Series(data_withna.astype(np.float32))

    def time_fletcher(self):
        self.fr_data.all()

    def time_fletcher_withna(self):
        self.fr_data_withna.all()

    def time_numpy(self):
        self.np_data.all()

    def time_numpy_withna(self):
        self.np_data_withna.all()
