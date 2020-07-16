"""Utility functions for the Knuth Moris Pratt string matching algorithm."""
import numpy as np

from fletcher._compat import njit


@njit(inline="always")
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


@njit(inline="always")
def append_to_kmp_matching(
    matched_len: int, character: int, pat: bytes, failure_function: np.ndarray
) -> int:
    """Append a character to a Knuth Moris Pratt matching.
    This function can be used to search for `pat` in a text with the KMP
    algorithm.
    Parameters
    ----------
    matched_len: int
        The length of the previous maximum prefix of `pat` that was a
        suffix of the text. Must sattisfy `0 <= matched_len < len(pat)`.
    character: int
        The next character of the text.
    pat: bytes
        The pattern that is searched in the text.
    failure_function: np.ndarray
        The KMP failure function of `pat`. Should be obtained through
        `compute_kmp_failure_function(pat)`.
    Returns
    -------
    int
        The length of the maximum prefix of `pat` that is a suffix of the
        text after appendng `character`. Always `=> 0` and
        `<= min(matched_len + 1, len(pat))`.
    """
    while matched_len > -1 and pat[matched_len] != character:
        matched_len = failure_function[matched_len]
    return matched_len + 1
