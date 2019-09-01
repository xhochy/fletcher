import numpy as np
import pandas as pd
import pyarrow as pa

import fletcher as fr


class BooleanAny:
    def setup(self):
        data = np.zeros(2 ** 24).astype(bool)
        self.fr_data = pd.Series(fr.FletcherArray(pa.array(data)))
        self.np_data = pd.Series(data.astype(np.float32))

    def time_fletcher(self):
        self.fr_data.any()

    def track_size_fletcher(self):
        return self.fr_data.nbytes

    def time_numpy(self):
        self.np_data.any()

    def track_size_numpy(self):
        return self.np_data.nbytes


class BooleanAll:
    def setup(self):
        data = np.ones(2 ** 24).astype(bool)
        self.fr_data = pd.Series(fr.FletcherArray(pa.array(data)))
        self.np_data = pd.Series(data.astype(np.float32))

    def time_fletcher(self):
        self.fr_data.all()

    def time_numpy(self):
        self.np_data.all()
