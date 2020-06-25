import warnings

import numba

try:
    import dask.dataframe  # noqa: F401
except ImportError:
    HAS_DASK = False
else:
    HAS_DASK = True


def njit(*args, **kws):
    """
    Equivalent to jit(nopython=True, nogil=True).

    See documentation for jit function/decorator for full description.
    """
    if "nopython" in kws:
        warnings.warn("nopython is set for njit and is ignored", RuntimeWarning)
    if "forceobj" in kws:
        warnings.warn("forceobj is set for njit and is ignored", RuntimeWarning)
    if "nogil" in kws:
        warnings.warn("nogil is set for njit and is ignored", RuntimeWarning)
    kws.update({"nopython": True, "nogil": True})
    return numba.jit(*args, **kws)
