import pandas as pd
import pandas_string as pd_str
import pyarrow as pa


class TimeSuite:
    def setup(self):
        array = [str(x) for x in range(2**15)]
        self.df = pd.DataFrame({'str': array})
        self.df_ext = pd.DataFrame({
            'str': pd_str.StringArray(pa.array(array))
        })

    def time_isnull(self):
        self.df['str'].isnull()
    
    def time_isnull_ext(self):
        self.df_ext['str'].isnull()
