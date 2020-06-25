import datetime

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pyarrow as pa


def test_basics(fletcher_array):
    df = pd.DataFrame(
        {
            "null": fletcher_array(pa.array([None, None], type=pa.null())),
            "bool": fletcher_array(pa.array([None, True], type=pa.bool_())),
            "int8": fletcher_array(pa.array([None, -1], type=pa.int8())),
            "uint8": fletcher_array(pa.array([None, 1], type=pa.uint8())),
            "int16": fletcher_array(pa.array([None, -1], type=pa.int16())),
            "uint16": fletcher_array(pa.array([None, 1], type=pa.uint16())),
            "int32": fletcher_array(pa.array([None, -1], type=pa.int32())),
            "uint32": fletcher_array(pa.array([None, 1], type=pa.uint32())),
            "int64": fletcher_array(pa.array([None, -1], type=pa.int64())),
            "uint64": fletcher_array(pa.array([None, 1], type=pa.uint64())),
            "float16": fletcher_array(
                pa.array([None, np.float16(-0.1)], type=pa.float16())
            ),
            "float32": fletcher_array(pa.array([None, -0.1], type=pa.float32())),
            "float64": fletcher_array(pa.array([None, -0.1], type=pa.float64())),
            "date32": fletcher_array(
                pa.array([None, datetime.date(2010, 9, 8)], type=pa.date32())
            ),
            "date64": fletcher_array(
                pa.array([None, datetime.date(2010, 9, 8)], type=pa.date64())
            ),
            # https://github.com/pandas-dev/pandas/issues/34986
            # "timestamp[s]": fletcher_array(
            #     pa.array(
            #         [None, datetime.datetime(2013, 12, 11, 10, 9, 8)],
            #         type=pa.timestamp("s"),
            #     )
            # ),
            # "timestamp[ms]": fletcher_array(
            #     pa.array(
            #         [None, datetime.datetime(2013, 12, 11, 10, 9, 8, 1000)],
            #         type=pa.timestamp("ms"),
            #     )
            # ),
            # "timestamp[us]": fletcher_array(
            #     pa.array(
            #         [None, datetime.datetime(2013, 12, 11, 10, 9, 8, 7)],
            #         type=pa.timestamp("us"),
            #     )
            # ),
            # FIXME: assert_extension_array_equal casts to numpy object thus cannot handle nanoseconds
            # 'timestamp[ns]': fletcher_array(pa.array([None, datetime.datetime(2013, 12, 11, 10, 9, 8, 7)], type=pa.timestamp("ns"))),
            "binary": fletcher_array(pa.array([None, b"122"], type=pa.binary())),
            "string": fletcher_array(pa.array([None, "🤔"], type=pa.string())),
            "duration[s]": fletcher_array(
                pa.array([None, datetime.timedelta(seconds=9)], type=pa.duration("s"))
            ),
            "duration[ms]": fletcher_array(
                pa.array(
                    [None, datetime.timedelta(milliseconds=8)], type=pa.duration("ms")
                )
            ),
            "duration[us]": fletcher_array(
                pa.array(
                    [None, datetime.timedelta(microseconds=7)], type=pa.duration("us")
                )
            ),
            # FIXME: assert_extension_array_equal casts to numpy object thus cannot handle nanoseconds
            # 'duration[ns]': fletcher_array(pa.array([None, datetime.timedelta(microseconds=7)], type=pa.duration("ns"))),
            "list[string]": fletcher_array(
                pa.array([None, [None, "🤔"]], type=pa.list_(pa.string()))
            ),
        }
    )
    ddf = dd.from_pandas(df, npartitions=2)

    meta_nonempty = ddf._meta_nonempty
    pdt.assert_frame_equal(meta_nonempty, df)

    result = ddf.compute()
    pdt.assert_frame_equal(result, df)
