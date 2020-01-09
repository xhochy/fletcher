import pandas.testing as tm
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import fletcher as fr


@pytest.mark.parametrize("continuous", [True, False])
def test_read_parquet(tmpdir, continuous):
    str_arr = pa.array(["a", None, "c"], pa.string())
    int_arr = pa.array([1, None, -2], pa.int32())
    bool_arr = pa.array([True, None, False], pa.bool_())
    table = pa.Table.from_arrays([str_arr, int_arr, bool_arr], ["str", "int", "bool"])

    pq.write_table(table, "df.parquet")
    result = fr.read_parquet("df.parquet", continuous=continuous)
    expected = fr.pandas_from_arrow(table, continuous=continuous)
    tm.assert_frame_equal(result, expected)
