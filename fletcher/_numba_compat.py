import warnings

import numba
import numpy as np

_string_buffer_types = np.uint8, np.uint32, np.uint8


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
