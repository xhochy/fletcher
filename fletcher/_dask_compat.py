from dask.dataframe.extensions import make_array_nonempty

from fletcher.base import FletcherChunkedDtype, FletcherContinuousDtype


@make_array_nonempty.register(FletcherChunkedDtype)
def _0(dtype):
    return dtype.example()


@make_array_nonempty.register(FletcherContinuousDtype)
def _1(dtype):
    return dtype.example()
