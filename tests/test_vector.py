import numba
import numpy as np
import pyarrow as pa

from fletcher.vector import (
    BitVector,
    ByteVector,
    StringArrayBuilder,
    finalize_string_array,
)


@numba.jit(nopython=True)
def append_range(bv):
    for i in range(2048):
        bv.append(8)


def test_allocate():
    bv = ByteVector(2)
    append_range(bv)


@numba.jit(nopython=True)
def build_bitarray_nopython():
    bv = BitVector(2)
    bv.append_true()
    bv.append_false()
    bv.append_false()
    bv.append_true()
    bv.append_true()
    bv.append_false()
    bv.append_true()
    return bv


@numba.jit
def build_stringarray(value1, value2, value3):
    sab = StringArrayBuilder(2)
    sab.append_value(value1, len(value1))
    sab.append_null()
    sab.append_value(value2, len(value2))
    sab.append_value(value3, len(value3))
    return sab


def test_stringvector():
    value1 = np.frombuffer(b"Test", dtype=np.uint8)
    value2 = np.frombuffer(b"", dtype=np.uint8)
    value3 = np.frombuffer("🤯".encode(), dtype=np.uint8)
    builder = build_stringarray(value1, value2, value3)
    arr = finalize_string_array(builder, pa.string())

    assert arr.to_pylist() == ["Test", None, "", "🤯"]
