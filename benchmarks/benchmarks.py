from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import six

import fletcher as fr


def generate_test_array(n):
    return [
        six.text_type(x) + six.text_type(x) + six.text_type(x) if x % 7 == 0 else None
        for x in range(n)
    ]


class TimeSuiteText:
    def setup(self):
        array = [
            ("a" * 50 + "b" if i % 2 == 0 else "c") * 5 + str(i) for i in range(2 ** 16)
        ]
        self.pattern = "a" * 30 + "b"

        self.df = pd.DataFrame({"str": array})
        self.df_ext = pd.DataFrame(
            {"str": fr.FletcherChunkedArray(pa.array(array, pa.string()))}
        )

    def time_count_no_regex(self):
        self.df["str"].str.count(self.pattern)

    def time_count_no_regex_ext(self):
        self.df_ext["str"].text.count(self.pattern, regex=False)

    def time_contains_no_regex(self):
        self.df["str"].str.contains(self.pattern, regex=False)

    def time_contains_no_regex_ext(self):
        self.df_ext["str"].text.contains(self.pattern, regex=False)

    def time_replace_no_regex(self):
        self.df["str"].str.replace(self.pattern, "bc")

    def time_replace_no_regex_ext(self):
        self.df_ext["str"].text.count(self.pattern, "bc", regex=False)


class TimeSuite:
    def setup(self):
        array = generate_test_array(2 ** 17)
        self.df = pd.DataFrame({"str": array})
        self.df_ext = pd.DataFrame(
            {"str": fr.FletcherChunkedArray(pa.array(array, pa.string()))}
        )

    def time_isnull(self):
        self.df["str"].isnull()

    def time_isnull_ext(self):
        self.df_ext["str"].isnull()

    def time_startswith(self):
        self.df["str"].str.startswith("10")

    def time_startswith_ext(self):
        self.df_ext["str"].text.startswith("10")

    def time_endswith(self):
        self.df["str"].str.endswith("10")

    def time_endswith_ext(self):
        self.df_ext["str"].text.endswith("10")

    def time_cat(self):
        self.df["str"].str.cat(self.df["str"])

    def time_cat_ext(self):
        self.df_ext["str"].text.cat(self.df_ext["str"])

    def time_zfill(self):
        self.df["str"].str.zfill(10)

    def time_zfill_ext(self):
        self.df_ext["str"].text.zfill(10)

    def time_contains_no_regex(self):
        self.df["str"].str.contains("1012102", regex=False)

    def time_contains_no_regex_ext(self):
        self.df_ext["str"].text.contains("1012102", regex=False)

    def time_contains_no_regex_ignore_case(self):
        self.df["str"].str.contains("0", regex=False, case=False)

    def time_contains_no_regex_ignore_case_ext(self):
        self.df_ext["str"].text.contains("0", regex=False, case=False)

    def time_contains_regex(self):
        self.df["str"].str.contains("[0-3]", regex=True)

    def time_contains_regex_ext(self):
        self.df_ext["str"].text.contains("[0-3]", regex=True)

    def time_contains_regex_ignore_case(self):
        self.df["str"].str.contains("[0-3]", regex=True, case=False)

    def time_contains_regex_ignore_case_ext(self):
        self.df_ext["str"].text.contains("[0-3]", regex=True, case=False)

    def time_concat(self):
        pd.concat([self.df["str"]] * 2)

    def time_concat_ext(self):
        pd.concat([self.df_ext["str"]] * 2)

    def time_count_no_regex(self):
        self.df["str"].str.count("001")

    def time_count_no_regex_ext(self):
        self.df_ext["str"].text.count("001", regex=False)

    def time_replace_no_regex(self):
        self.df["str"].str.replace("001", "23")

    def time_replace_no_regex_ext(self):
        self.df_ext["str"].text.count("001", "23", regex=False)


class Indexing(object):
    # index and value have diverse values, disable type checks for them
    indexer: Any
    value: Any

    n = 2 ** 12

    params = [
        (True, False),
        ("scalar_value", "array_value"),
        ("int", "int_array", "bool_array", "slice"),
    ]
    param_names = ["chunked", "values", "indices"]

    def setup(self, chunked, value, indices):
        # assert np.isscalar(values) or len(values) == len(indices)
        array = generate_test_array(self.n)
        if indices == "int":
            if value == "array_value":
                raise NotImplementedError()
            self.indexer = 50
        elif indices == "int_array":
            self.indexer = list(range(0, self.n, 5))
        elif indices == "bool_array":
            self.indexer = np.zeros(self.n, dtype=bool)
            self.indexer[list(range(0, self.n, 5))] = True
        elif indices == "slice":
            self.indexer = slice(0, self.n, 5)

        if value == "scalar_value":
            self.value = "setitem"
        elif value == "array_value":
            self.value = [str(x) for x in range(self.n)]
            self.value = np.array(self.value)[self.indexer]
            if len(self.value) == 1:
                self.value = self.value[0]

        self.df = pd.DataFrame({"str": array})
        if chunked:
            array = np.array_split(array, 1000)
        else:
            array = [array]
        self.df_ext = pd.DataFrame(
            {
                "str": fr.FletcherChunkedArray(
                    pa.chunked_array([pa.array(chunk, pa.string()) for chunk in array])
                )
            }
        )

    def time_getitem(self, chunked, value, indices):
        self.df_ext["str"][self.indexer]

    def time_getitem_obj(self, chunked, value, indices):
        self.df["str"][self.indexer]

    def time_setitem(self, chunked, value, indices):
        self.df_ext["str"][self.indexer] = self.value

    def time_setitem_obj(self, chunked, value, indices):
        self.df["str"][self.indexer] = self.value
