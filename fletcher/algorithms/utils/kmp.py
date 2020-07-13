"""Utility functions for the Knuth Moris Pratt string matching algorithm."""
import numpy as np

from fletcher._compat import njit


@njit
def compute_kmp_failure_function(pat: bytes) -> np.ndarray:
    """Compute the Knuth Moris Pratt failure function.

    Parameters
    ----------
    pat : bytes
        The bytes representation of the string for which to compute the KMP
        failure function.

    Returns
    -------
    Numpy array f of len(pat) + 1 integers.
        f[0] = -1. For i > 0, f[i] is equal to the length of the longest
        propper suffix of pat[:i] that is a prefix of pat. Since, only propper
        suffixes are considered, for i > 0 we have 0 <= f[i] < i.

    Examples
    --------
    >>> comp


    """

    length = len(pat)
    f = np.empty(length + 1, dtype=np.int32)

    f[0] = -1
    for i in range(1, length + 1):
        f[i] = f[i - 1]
        while f[i] != -1 and pat[f[i]] != pat[i - 1]:
            f[i] = f[f[i]]
        f[i] += 1

    return f

@njit
def append_to_kmp_matching(
    matched_len: int,
    character: int,
    pat: bytes,
    failure_function: np.ndarray
) -> np.ndarray:
    while matched_len > -1 and pat[matched_len] != character:
        matched_len = failure_function[matched_len]
    return matched_len + 1
