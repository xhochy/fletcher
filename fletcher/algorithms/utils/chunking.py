"""Utility functions to deal with chunked arrays."""

from functools import singledispatch, wraps
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pyarrow as pa


def _calculate_chunk_offsets(chunked_array: pa.ChunkedArray) -> np.ndarray:
    """Return an array holding the indices pointing to the first element of each chunk."""
    offset = 0
    offsets = []
    for chunk in chunked_array.iterchunks():
        offsets.append(offset)
        offset += len(chunk)
    return np.array(offsets)


def _in_chunk_offsets(
    arr: pa.ChunkedArray, offsets: List[int]
) -> List[Tuple[int, int, int]]:
    """Calculate the access ranges for a given list of offsets.

    All chunk start indices must be included as offsets and the offsets must be
    unique.

    Returns a list of tuples that contain:
     * The index of the given chunk
     * The position inside the chunk
     * The length of the current range
    """
    new_offsets = []
    pos = 0
    chunk = 0
    chunk_pos = 0
    for offset, offset_next in zip(offsets, offsets[1:] + [len(arr)]):
        diff = offset - pos
        chunk_remains = len(arr.chunk(chunk)) - chunk_pos
        step = offset_next - offset
        if diff == 0:  # The first offset
            new_offsets.append((chunk, chunk_pos, step))
        elif diff == chunk_remains:
            chunk += 1
            chunk_pos = 0
            pos += chunk_remains
            new_offsets.append((chunk, chunk_pos, step))
        else:  # diff < chunk_remains
            chunk_pos += diff
            pos += diff
            new_offsets.append((chunk, chunk_pos, step))
    return new_offsets


def _combined_in_chunk_offsets(
    a: pa.ChunkedArray, b: pa.ChunkedArray
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    offsets_a = _calculate_chunk_offsets(a)
    offsets_b = _calculate_chunk_offsets(b)
    offsets = sorted(set(list(offsets_a) + list(offsets_b)))
    in_a_offsets = _in_chunk_offsets(a, offsets)
    in_b_offsets = _in_chunk_offsets(b, offsets)
    return in_a_offsets, in_b_offsets


def apply_per_chunk(func):
    """Apply a function to each chunk if the input is chunked."""

    @wraps(func)
    def wrapper(arr: Union[pa.Array, pa.ChunkedArray], *args, **kwargs):
        if isinstance(arr, pa.ChunkedArray):
            return pa.chunked_array(
                [func(chunk, *args, **kwargs) for chunk in arr.chunks]
            )
        else:
            return func(arr, *args, **kwargs)

    return wrapper


def _not_implemented_path(*args, **kwargs):
    raise NotImplementedError("Dispatching path not implemented")


@singledispatch
def dispatch_chunked_binary_map(a: Any, b: Any, ops: Dict[str, Callable]):
    """
    Apply a map-like binary function where at least one of the arguments is an Arrow structure.

    This will yield a pyarrow.Arrow or pyarrow.ChunkedArray as an output.

    Parameters
    ----------
    a: scalar or np.ndarray or pa.Array or pa.ChunkedArray
    b: scalar or np.ndarray or pa.Array or pa.ChunkedArray
    op: dict
        Dictionary with the keys ('array_array', 'array_nparray', 'nparray_array',
        'array_scalar', 'scalar_array')
    """
    # a is neither a pa.Array nor a pa.ChunkedArray, we expect only numpy.ndarray or scalars.
    if isinstance(b, pa.ChunkedArray):
        if np.isscalar(a):
            new_chunks = []
            for chunk in b.iterchunks():
                new_chunks.append(dispatch_chunked_binary_map(a, chunk, ops))
            return pa.chunked_array(new_chunks)
        else:
            if len(a) != len(b):
                raise ValueError("Inputs don't have the same length.")
            new_chunks = []
            offsets = _calculate_chunk_offsets(b)
            for chunk, offset in zip(b.iterchunks(), offsets):
                new_chunks.append(
                    dispatch_chunked_binary_map(
                        a[offset : offset + len(chunk)], chunk, ops
                    )
                )
            return pa.chunked_array(new_chunks)
    elif isinstance(b, pa.Array):
        if np.isscalar(a):
            return ops.get("scalar_array", _not_implemented_path)(a, b)
        else:
            return ops.get("nparray_array", _not_implemented_path)(a, b)
    else:
        # Should never be reached, add a safe-guard
        raise NotImplementedError(f"Cannot apply ufunc on {type(a)} and {type(b)}")


@dispatch_chunked_binary_map.register(pa.ChunkedArray)
def _1(a: pa.ChunkedArray, b: Any, ops: Dict[str, Callable]):
    """Apply a NumPy ufunc where at least one of the arguments is an Arrow structure."""
    if isinstance(b, pa.ChunkedArray):
        if len(a) != len(b):
            raise ValueError("Inputs don't have the same length.")
        in_a_offsets, in_b_offsets = _combined_in_chunk_offsets(a, b)

        new_chunks: List[pa.Array] = []
        for a_offset, b_offset in zip(in_a_offsets, in_b_offsets):
            a_slice = a.chunk(a_offset[0])[a_offset[1] : a_offset[1] + a_offset[2]]
            b_slice = b.chunk(b_offset[0])[b_offset[1] : b_offset[1] + b_offset[2]]
            new_chunks.append(dispatch_chunked_binary_map(a_slice, b_slice, ops))
        return pa.chunked_array(new_chunks)
    elif np.isscalar(b):
        new_chunks = []
        for chunk in a.iterchunks():
            new_chunks.append(dispatch_chunked_binary_map(chunk, b, ops))
        return pa.chunked_array(new_chunks)
    else:
        if len(a) != len(b):
            raise ValueError("Inputs don't have the same length.")
        new_chunks = []
        offsets = _calculate_chunk_offsets(a)
        for chunk, offset in zip(a.iterchunks(), offsets):
            new_chunks.append(
                dispatch_chunked_binary_map(chunk, b[offset : offset + len(chunk)], ops)
            )
        return pa.chunked_array(new_chunks)


@dispatch_chunked_binary_map.register(pa.Array)
def _2(a: pa.Array, b: Any, ops: Dict[str, Callable]):
    """Apply a NumPy ufunc where at least one of the arguments is an Arrow structure."""
    if isinstance(b, pa.ChunkedArray):
        if len(a) != len(b):
            raise ValueError("Inputs don't have the same length.")
        new_chunks = []
        offsets = _calculate_chunk_offsets(b)
        for chunk, offset in zip(b.iterchunks(), offsets):
            new_chunks.append(
                dispatch_chunked_binary_map(a[offset : offset + len(chunk)], chunk, ops)
            )
        return pa.chunked_array(new_chunks)
    elif isinstance(b, pa.Array):
        if len(a) != len(b):
            raise ValueError("Inputs don't have the same length.")
        return ops.get("array_array", _not_implemented_path)(a, b)
    else:
        if np.isscalar(b):
            return ops.get("array_scalar", _not_implemented_path)(a, b)
        else:
            if len(a) != len(b):
                raise ValueError("Inputs don't have the same length.")
            return ops.get("array_nparray", _not_implemented_path)(a, b)
