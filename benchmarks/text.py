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


class TimeSuitePatterns:
    """
    Special benchmark suite for string algorithms looking for patterns.
    """

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
        self.df_ext["str"].str.count(self.pattern)

    def time_count_no_regex_ext_fr(self):
        self.df_ext["str"].fr_str.count(self.pattern, regex=False)

    def time_contains_no_regex(self):
        self.df["str"].str.contains(self.pattern, regex=False)

    def time_contains_no_regex_ext(self):
        self.df_ext["str"].str.contains(self.pattern, regex=False)

    def time_contains_no_regex_ext_fr(self):
        self.df_ext["str"].fr_str.contains(self.pattern, regex=False)

    def time_replace_no_regex(self):
        self.df["str"].str.replace(self.pattern, "bc")

    def time_replace_no_regex_ext(self):
        self.df_ext["str"].str.replace(self.pattern, "bc", regex=False)

    def time_replace_no_regex_ext_fr(self):
        self.df_ext["str"].fr_str.replace(self.pattern, "bc", regex=False)


class TimeSuiteText:
    def setup(self):
        array = generate_test_array(2 ** 17)
        self.df = pd.DataFrame({"str": array})
        self.df_ext = pd.DataFrame(
            {"str": fr.FletcherChunkedArray(pa.array(array, pa.string()))}
        )

    ###########################
    ## General functionality ##
    ###########################

    def time_isnull(self):
        self.df["str"].isnull()

    def time_isnull_ext(self):
        self.df_ext["str"].isnull()

    def time_concat(self):
        pd.concat([self.df["str"]] * 2)

    def time_concat_ext(self):
        pd.concat([self.df_ext["str"]] * 2)

    ###############################
    ## String accessor functions ##
    ###############################

    def time_capitalize(self):
        self.df["str"].str.capitalize()

    def time_capitalize_ext(self):
        self.df_ext["str"].str.capitalize()

    def time_capitalize_ext_fr(self):
        self.df_ext["str"].fr_str.capitalize()

    def time_casefold(self):
        self.df["str"].str.casefold()

    def time_casefold_ext(self):
        self.df_ext["str"].str.casefold()

    def time_casefold_ext_fr(self):
        self.df_ext["str"].fr_str.casefold()

    def time_cat(self):
        self.df["str"].str.cat(self.df["str"])

    def time_cat_ext(self):
        self.df_ext["str"].str.cat(self.df["str"])

    def time_cat_ext_fr(self):
        self.df_ext["str"].fr_str.cat(self.df_ext["str"])

    def time_contains_no_regex(self):
        self.df["str"].str.contains("1012102", regex=False)

    def time_contains_no_regex_ext(self):
        self.df_ext["str"].str.contains("1012102", regex=False)

    def time_contains_no_regex_ext_fr(self):
        self.df_ext["str"].fr_str.contains("1012102", regex=False)

    def time_contains_no_regex_ignore_case(self):
        self.df["str"].str.contains("0", regex=False, case=False)

    def time_contains_no_regex_ignore_case_ext(self):
        self.df_ext["str"].str.contains("0", regex=False, case=False)

    def time_contains_no_regex_ignore_case_ext_fr(self):
        self.df_ext["str"].fr_str.contains("0", regex=False, case=False)

    def time_contains_regex(self):
        self.df["str"].str.contains("[0-3]", regex=True)

    def time_contains_regex_ext(self):
        self.df_ext["str"].str.contains("[0-3]", regex=True)

    def time_contains_regex_ext_fr(self):
        self.df_ext["str"].fr_str.contains("[0-3]", regex=True)

    def time_contains_regex_ignore_case(self):
        self.df["str"].str.contains("[0-3]", regex=True, case=False)

    def time_contains_regex_ignore_case_ext(self):
        self.df_ext["str"].str.contains("[0-3]", regex=True, case=False)

    def time_contains_regex_ignore_case_ext_fr(self):
        self.df_ext["str"].fr_str.contains("[0-3]", regex=True, case=False)

    def time_count_no_regex(self):
        self.df["str"].str.count("001")

    def time_count_no_regex_ext(self):
        self.df_ext["str"].str.count("001")

    def time_count_no_regex_ext_fr(self):
        self.df_ext["str"].fr_str.count("001", regex=False)

    def time_count_regex(self):
        self.df["str"].str.count("[0-3]")

    def time_count_regex_ext(self):
        self.df_ext["str"].str.count("[0-3]")

    def time_count_regex_ext_fr(self):
        self.df_ext["str"].fr_str.count("[0-3]", regex=True)

    def time_endswith(self):
        self.df["str"].str.endswith("10")

    def time_endswith_ext(self):
        self.df_ext["str"].str.endswith("10")

    def time_endswith_ext_fr(self):
        self.df_ext["str"].fr_str.endswith("10")

    def time_extract(self):
        self.df["str"].str.extract("([0-3]+)")

    def time_extract_ext(self):
        self.df_ext["str"].str.extract("([0-3]+)")

    def time_extract_ext_fr(self):
        self.df_ext["str"].fr_str.extract("([0-3]+)")

    def time_extractall(self):
        self.df["str"].str.extractall("([0-3]+)")

    def time_extractall_ext(self):
        self.df_ext["str"].str.extractall("([0-3]+)")

    def time_extractall_ext_fr(self):
        self.df_ext["str"].fr_str.extractall("([0-3]+)")

    def time_find(self):
        self.df["str"].str.find("10")

    def time_find_ext(self):
        self.df_ext["str"].str.find("10")

    def time_find_ext_fr(self):
        self.df_ext["str"].fr_str.find("10")

    def time_findall(self):
        self.df["str"].str.findall("([0-3]+)")

    def time_findall_ext(self):
        self.df_ext["str"].str.findall("([0-3]+)")

    def time_findall_ext_fr(self):
        self.df_ext["str"].fr_str.findall("([0-3]+)")

    def time_get(self):
        self.df["str"].str.get(2)

    def time_get_ext(self):
        self.df_ext["str"].str.get(2)

    def time_get_ext_fr(self):
        self.df_ext["str"].fr_str.get(2)

    # We don't benchmark index as it is the same as find but just raising errors instead of returning -1

    def time_len(self):
        self.df["str"].str.len()

    def time_len_ext(self):
        self.df_ext["str"].str.len()

    def time_len_ext_fr(self):
        self.df_ext["str"].fr_str.len()

    def time_ljust(self):
        self.df["str"].str.ljust(10)

    def time_ljust_ext(self):
        self.df_ext["str"].str.ljust(10)

    def time_ljust_ext_fr(self):
        self.df_ext["str"].fr_str.ljust(10)

    def time_lower(self):
        self.df["str"].str.lower()

    def time_lower_ext(self):
        self.df_ext["str"].str.lower()

    def time_lower_ext_fr(self):
        self.df_ext["str"].fr_str.lower()

    def time_lstrip(self):
        self.df["str"].str.lstrip("0")

    def time_lstrip_ext(self):
        self.df_ext["str"].str.lstrip("0")

    def time_lstrip_ext_fr(self):
        self.df_ext["str"].fr_str.lstrip("0")

    def time_match(self):
        self.df["str"].str.match("([0-3]+)")

    def time_match_ext(self):
        self.df_ext["str"].str.match("([0-3]+)")

    def time_match_ext_fr(self):
        self.df_ext["str"].fr_str.match("([0-3]+)")

    def time_normalize(self):
        self.df["str"].str.normalize(form="NFC")

    def time_normalize_ext(self):
        self.df_ext["str"].str.normalize(form="NFC")

    def time_normalize_ext_fr(self):
        self.df_ext["str"].fr_str.normalize(form="NFC")

    # We skip pad as it is just center/rjust/ljust

    def time_partition(self):
        self.df["str"].str.partition("0")

    # Not supported yet, see the corresponding unit test
    # def time_partition_ext(self):
    #     self.df_ext["str"].str.partition("0")

    # Not supported yet, see the corresponding unit test
    # def time_partition_ext_fr(self):
    #    self.df_ext["str"].fr_str.partition("0")

    def time_repeat(self):
        self.df["str"].str.repeat(3)

    def time_repeat_ext(self):
        self.df_ext["str"].str.repeat(3)

    def time_repeat_ext_fr(self):
        self.df_ext["str"].fr_str.repeat(3)

    def time_replace_no_regex(self):
        self.df["str"].str.replace("001", "23", regex=False)

    def time_replace_no_regex_ext(self):
        self.df_ext["str"].str.replace("001", "23", regex=False)

    def time_replace_no_regex_ext_fr(self):
        self.df_ext["str"].fr_str.replace("001", "23", regex=False)

    def time_replace_regex(self):
        self.df["str"].str.replace("[0-3]", "5", regex=True)

    def time_replace_regex_ext(self):
        self.df_ext["str"].str.replace("[0-3]", "5", regex=True)

    def time_replace_regex_ext_fr(self):
        self.df_ext["str"].fr_str.replace("[0-3]", "5", regex=False)

    def time_rfind(self):
        self.df["str"].str.rfind("10")

    def time_rfind_ext(self):
        self.df_ext["str"].str.rfind("10")

    def time_rfind_ext_fr(self):
        self.df_ext["str"].fr_str.rfind("10")

    def time_rjust(self):
        self.df["str"].str.rjust(10)

    def time_rjust_ext(self):
        self.df_ext["str"].str.rjust(10)

    def time_rjust_ext_fr(self):
        self.df_ext["str"].fr_str.rjust(10)

    def time_rpartition(self):
        self.df["str"].str.rpartition("0")

    # Not supported yet, see the corresponding unit test
    # def time_rpartition_ext(self):
    #     self.df_ext["str"].str.rpartition("0")

    # Not supported yet, see the corresponding unit test
    # def time_rpartition_ext_fr(self):
    #    self.df_ext["str"].fr_str.rpartition("0")

    def time_rstrip(self):
        self.df["str"].str.rstrip("0")

    def time_rstrip_ext(self):
        self.df_ext["str"].str.rstrip("0")

    def time_rstrip_ext_fr(self):
        self.df_ext["str"].fr_str.rstrip("0")

    def time_slice(self):
        self.df["str"].str.slice(2, 5)

    def time_slice_ext(self):
        self.df_ext["str"].str.slice(2, 5)

    # FIXME: Timeouts
    # def time_slice_ext_fr(self):
    #     self.df_ext["str"].fr_str.slice(2, 5)

    def time_slice_step2(self):
        self.df["str"].str.slice(2, 5, 2)

    def time_slice_step2_ext(self):
        self.df_ext["str"].str.slice(2, 5, 2)

    # FIXME: Timeouts
    # def time_slice_step2_ext_fr(self):
    #     self.df_ext["str"].fr_str.slice(2, 5, 2)

    def time_slice_replace(self):
        self.df["str"].str.slice_replace(start=1, stop=3, repl="X")

    def time_slice_replace_ext(self):
        self.df_ext["str"].str.slice_replace(start=1, stop=3, repl="X")

    def time_slice_replace_ext_fr(self):
        self.df_ext["str"].fr_str.slice_replace(start=1, stop=3, repl="X")

    def time_split(self):
        self.df["str"].str.split("0")

    def time_split_ext(self):
        self.df_ext["str"].str.split("0")

    def time_split_ext_fr(self):
        self.df_ext["str"].fr_str.split("0")

    def time_rsplit(self):
        self.df["str"].str.rsplit("0")

    def time_rsplit_ext(self):
        self.df_ext["str"].str.split("0")

    def time_rsplit_ext_fr(self):
        self.df_ext["str"].fr_str.rsplit("0")

    def time_startswith(self):
        self.df["str"].str.startswith("10")

    def time_startswith_ext(self):
        self.df_ext["str"].str.startswith("10")

    def time_startswith_ext_fr(self):
        self.df_ext["str"].fr_str.startswith("10")

    def time_strip(self):
        self.df["str"].str.strip("0")

    def time_strip_ext(self):
        self.df_ext["str"].str.strip("0")

    # FIXME: timeouts
    # def time_strip_ext_fr(self):
    #     self.df_ext["str"].fr_str.strip('0')

    def time_swapcase(self):
        self.df["str"].str.swapcase()

    def time_swapcase_ext(self):
        self.df_ext["str"].str.swapcase()

    def time_swapcase_ext_fr(self):
        self.df_ext["str"].fr_str.swapcase()

    def time_title(self):
        self.df["str"].str.title()

    def time_title_ext(self):
        self.df_ext["str"].str.title()

    def time_title_ext_fr(self):
        self.df_ext["str"].fr_str.title()

    def time_translate(self):
        self.df["str"].str.translate({"0": "9", "1": "8", "2": "7", "3": "6", "4": "5"})

    def time_translate_ext(self):
        self.df_ext["str"].str.translate(
            {"0": "9", "1": "8", "2": "7", "3": "6", "4": "5"}
        )

    def time_translate_ext_fr(self):
        self.df_ext["str"].fr_str.translate(
            {"0": "9", "1": "8", "2": "7", "3": "6", "4": "5"}
        )

    def time_upper(self):
        self.df["str"].str.upper()

    def time_upper_ext(self):
        self.df_ext["str"].str.upper()

    def time_upper_ext_fr(self):
        self.df_ext["str"].fr_str.upper()

    def time_wrap(self):
        self.df["str"].str.wrap(5)

    def time_wrap_ext(self):
        self.df_ext["str"].str.wrap(5)

    def time_wrap_ext_fr(self):
        self.df_ext["str"].fr_str.wrap(5)

    def time_zfill(self):
        self.df["str"].str.zfill(10)

    def time_zfill_ext(self):
        self.df_ext["str"].str.zfill(10)

    def time_zfill_ext_fr(self):
        self.df_ext["str"].fr_str.zfill(10)

    def time_isalnum(self):
        self.df["str"].str.isalnum()

    def time_isalnum_ext(self):
        self.df_ext["str"].str.isalnum()

    def time_isalnum_ext_fr(self):
        self.df_ext["str"].fr_str.isalnum()

    def time_isalpha(self):
        self.df["str"].str.isalpha()

    def time_isalpha_ext(self):
        self.df_ext["str"].str.isalpha()

    def time_isalpha_ext_fr(self):
        self.df_ext["str"].fr_str.isalpha()

    def time_isdigit(self):
        self.df["str"].str.isdigit()

    def time_isdigit_ext(self):
        self.df_ext["str"].str.isdigit()

    def time_isdigit_ext_fr(self):
        self.df_ext["str"].fr_str.isdigit()

    def time_isspace(self):
        self.df["str"].str.isspace()

    def time_isspace_ext(self):
        self.df_ext["str"].str.isspace()

    def time_isspace_ext_fr(self):
        self.df_ext["str"].fr_str.isspace()

    def time_islower(self):
        self.df["str"].str.islower()

    def time_islower_ext(self):
        self.df_ext["str"].str.islower()

    def time_islower_ext_fr(self):
        self.df_ext["str"].fr_str.islower()

    def time_isupper(self):
        self.df["str"].str.isupper()

    def time_isupper_ext(self):
        self.df_ext["str"].str.isupper()

    def time_isupper_ext_fr(self):
        self.df_ext["str"].fr_str.isupper()

    def time_istitle(self):
        self.df["str"].str.istitle()

    def time_istitle_ext(self):
        self.df_ext["str"].str.istitle()

    def time_istitle_ext_fr(self):
        self.df_ext["str"].fr_str.istitle()

    def time_isnumeric(self):
        self.df["str"].str.isnumeric()

    def time_isnumeric_ext(self):
        self.df_ext["str"].str.isnumeric()

    def time_isnumeric_ext_fr(self):
        self.df_ext["str"].fr_str.isnumeric()

    def time_isdecimal(self):
        self.df["str"].str.isdecimal()

    def time_isdecimal_ext(self):
        self.df_ext["str"].str.isdecimal()

    def time_isdecimal_ext_fr(self):
        self.df_ext["str"].fr_str.isdecimal()


class Dummies:
    def setup(self):
        self.s = pd.Series(pd._testing.makeStringIndex(10 ** 5)).str.join("|")
        self.s_ext = pd.Series(
            fr.FletcherChunkedArray(pa.array(self.s.to_numpy(), pa.string()))
        )

    def time_get_dummies(self):
        self.s.str.get_dummies("|")

    def time_get_dummies_ext(self):
        self.s_ext.str.get_dummies("|")

    def time_get_dummies_ext_fr(self):
        self.s_ext.fr_str.get_dummies("|")


class Indexing:
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
