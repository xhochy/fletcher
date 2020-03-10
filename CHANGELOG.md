Changelog
=========

0.3.1
-----

* Support roundtrips of `pandas.DataFrame` instances with `fletcher` columns through `pyarrow` data structures.
* Move CI to Github Actions

0.3.0
-----

Major changes:
 * We now provide two different extension array implementations.
   There now is the more simpler `FletcherContinuousArray` which is backed by a `pyarrow.Array` instance and thus is always a continuous memory segments.
   The initial `FletcherArray` which is backed by a `pyarrow.ChunkedArray` is now renamed to `FletcherChunkedArray`.
   While `pyarrow.ChunkedArray` allows for more flexibility on how the data is stored, the implementation of algorithms is more complex for it.
   As this hinders contributions and also the adoption in downstream libraries, we now provide both implementations with an equal level of support.
   We don't provide the more general named class `FletcherArray` anymore as there is not a clear opinion on whether this should point to `FletcherContinuousArray` or `FletcherChunkedArray`.
   As usage increases, we might provide such an alias class in future again.
 * Support for `ArithmeticOps` and `ComparisonOps` on numerical data as well as numeric reductions such as `sum`.
   This should allow the use of nullable int and float type for many use cases.
   Performance of nullable integeter columns is on the same level as in `pandas.IntegerArray` as we have similar implementations of the masked arithmetic.
   In future versions, we plan to delegate the workload into the C++ code of `pyarrow` and expect significant performance improvements though the usage of bitmasks over bytemasks.
 * `any` and `all` are now efficiently implemented on boolean arrays.
   We [blogged about this](https://uwekorn.com/2019/09/02/boolean-array-with-missings.html) and how its performance is about twice as fast while only using 1/16 - 1/32 of RAM as the reference boolean array with missing in `pandas`.
   This is due to the fact that prior to `pandas=1.0` you have had to use a float array to have a boolean array that can deal with missing values.
   In `pandas=1.0` a new `BooleanArray` class was added that improves this stituation but also change a bit of the logic.
   We will adapt to this class in the next release and also publish new benchmarks.

New features / performance improvements:
 * For `FletcherContinuousArray` in general and all `FletcherChunkedArray` instances with a single chunk, we now provide an efficient implementation of `take`.
 * Support for Python 3.8 and Pandas 1.0
 * We now check typing in CI using `mypy` and have annotated the code with type hints.
   We only plan to mark the packages as `py.typed` when `pandas` is also marked as `py.typed`.
 * You can query `fletcher` for its version via `fletcher.__version__`
 * Implemented `.str.cat` as `.fr_text.cat` for arrays with `pa.string()` dtype.
 * `unique` is now supported on all array types where `pyarrow` provides a `unique` implementation.

0.2.0
-----

 * Drop Python 2 support
 * Support for Python 3.7
 * Fixed handling of `date` columns due to new default behaviours in `pyarrow`.

0.1.2
-----

Rerelease with the sole purpose of rendering MarkDown on PyPI.

0.1.1
-----

Load the README in setup.py to have a description on PyPI.

0.1.0
-----

Initial release of fletcher that is based on Pandas 0.23.3 and Apache Arrow 0.9.
This release already supports any Apache Arrow type but the unit tests are yet
limited to string and date.
