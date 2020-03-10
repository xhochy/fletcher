import pandas as pd
import pandas.testing as pdt
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fletcher import FletcherChunkedArray, FletcherContinuousArray


@pytest.mark.parametrize("array_type", [FletcherContinuousArray, FletcherChunkedArray])
def test_arrow_roundtrip(array_type):
    df = pd.DataFrame({"col": array_type(["A", "B"])})
    df_round = pa.Table.from_pandas(df).to_pandas()
    pdt.assert_frame_equal(df, df_round)


@pytest.mark.parametrize("array_type", [FletcherContinuousArray, FletcherChunkedArray])
def test_parquet_roundtrip(array_type):
    df = pd.DataFrame({"col": array_type(["A", "B"])})
    table = pa.Table.from_pandas(df)
    buf = pa.BufferOutputStream()
    pq.write_table(table, buf)
    reader = pa.BufferReader(buf.getvalue().to_pybytes())
    table = pq.read_table(reader)
    pdt.assert_frame_equal(df, table.to_pandas())
