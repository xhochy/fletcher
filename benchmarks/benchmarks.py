import pandas as pd
import pandas_string as pd_str
import pyarrow as pa


class TimeSuite:
    def setup(self):
        array = [
            str(x) + str(x) + str(x) if x % 7 == 0 else None
            for x in range(2**15)
        ]
        self.df = pd.DataFrame({'str': array})
        self.df_ext = pd.DataFrame({
            'str': pd_str.StringArray(pa.array(array))
        })

        # make sure numba compilation is not included
        self.df_ext['str'].isnull()
        self.df_ext['str'].text.startswith('')
        self.df_ext['str'].text.endswith('')

    def time_isnull(self):
        for _ in range(10):
            self.df['str'].isnull()

    def time_isnull_ext(self):
        for _ in range(10):
            self.df_ext['str'].isnull()

    def time_startswith(self):
        for _ in range(10):
            self.df['str'].str.startswith('10')

    def time_startswith_ext(self):
        for _ in range(10):
            self.df_ext['str'].text.startswith('10')

    def time_endswith(self):
        for _ in range(10):
            self.df['str'].str.endswith('10')

    def time_endswith_ext(self):
        for _ in range(10):
            self.df_ext['str'].text.endswith('10')
