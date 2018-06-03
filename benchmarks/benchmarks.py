import fletcher as fr
import numpy as np
import pandas as pd
import pyarrow as pa


class TimeStringSuite(object):

    def setup(self):
        array = [
            str(x) + str(x) + str(x) if x % 7 != 0 else None for x in range(2 ** 20)
        ]
        self.df = pd.DataFrame({"str": array, "int": np.arange(len(array))})
        self.df_ext = pd.DataFrame(
            {"str": fr.StringArray(pa.array(array)), "int": np.arange(len(array))}
        )

    def time_isnull(self):
        self.df["str"].isnull()

    def time_isnull_ext(self):
        self.df_ext["str"].isnull()

    def time_startswith(self):
        self.df["str"].str.startswith("10")

    def time_startswith_ext(self):
        self.df_ext["str"].text.startswith("10")

    def time_startswith_na(self):
        self.df["str"].str.startswith("10", na=False)

    def time_startswith_na_ext(self):
        self.df_ext["str"].text.startswith("10", na=False)

    def time_endswith_na(self):
        self.df["str"].str.endswith("10", na=False)

    def time_endswith_na_ext(self):
        self.df_ext["str"].text.endswith("10", na=False)

    def time_groupby(self):
        self.df.groupby("str").sum()

    def time_groupby_ext(self):
        self.df_ext.groupby("str").sum()
