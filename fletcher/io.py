from typing import List, Optional

import pandas as pd
import pyarrow.parquet as pq

from fletcher.base import pandas_from_arrow


def read_parquet(
    path, columns: Optional[List[str]] = None, continuous: bool = False
) -> pd.DataFrame:
    """
    Load a parquet object from the file path, returning a DataFrame with fletcher columns.

    Parameters
    ----------
    path : str or file-like
    continuous : bool
        Use FletcherContinuousArray instead of FletcherChunkedArray

    Returns
    -------
    pd.DataFrame
    """
    table = pq.read_table(path, columns=columns)
    return pandas_from_arrow(table, continuous=continuous)
