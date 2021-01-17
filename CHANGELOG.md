Changelog
=========

Starting with 0.5, we will follow the following versioning scheme:

* We don't bump MAJOR yet.
* We bump MINOR on breaking changes.
* We increase PATCH otherwise.

0.7.2 (2021-01-17)
------------------

* Allow NumPy ufuncs to work with `np.ndarray` outputs where operations are clearly defined (i.e. the fletcher array has no nulls).

0.7.1 (2020-12-29)
------------------

* Fix return values for `str` functions with `pandas=1.2` and `pyarrow=1`.
* Ensure that parallel variants of `apply_binary_str` actually parallize.

0.7.0 (2020-12-07)
------------------

* Add tests for all `str` functions.
* Fix tests for `pyarrow=0.17.1` and add CI jobs for `0.17.1` and `1.0.1`.
* Implement a faster take for list arrays.
* Use `utf8_is_*` functions from Apache Arrow if available.
* Simplify `factorize` implementation to work for chunked arrays with more or less than a single chunk.
* Switch to `pandas.NA` as the user-facing null value
* Add convenience function `fletcher.algorithms.string.apply_binary_str` to apply a binary function on two string columns.

0.6.2 (2020-10-20)
------------------

* Return correct index in functions like `fr_str.extractall`.

0.6.1 (2020-10-14)
------------------

* Create a shallow copy on `.astype(equal dtype, copy=True)`.
* Import `pad_1d` only in older `pandas` versions, otherwise use `get_fill_func`
* Handle `fr_str.extractall` and similar functions correctly, returning a `pd.Dataframe` containing accoring `fletcher` array types.

0.6.0 (2020-09-23)
------------------

* Use `binary_contains_exact` if available from `pyarrow` instead of our own numba-based implementation.
* Provide two more consistent accessors:
 * `.fr_strx`: Call efficient string functions on `fletcher` arrays, error if not available.
 * `.fr_str`: Call string functions on `fletcher` and `object`-typed arrays, convert to `object` if no `fletcher` function is available.
* Add a numba-based implementation for `strip`, `slice`, and `replace`.
* Support `LargeListArray` as a backing structure for lists.
* Implement `isnan` ufunc.

0.5.1 (2020-09-21)
------------------

* Release the GIL where possible.
* Register with dask's `make_array_nonempty` to be able to handle the extension types in `dask`.

0.5.0 (2020-06-23)
------------------

* Implement `FletcherBaseArray.__or__` and `FletcherBaseArray.__any__` to support `pandas.Series.replace`.

0.4.0 (2020-06-16)
------------------

* Forward the `__array__` protocol directly to Arrow
* Add naive implementation for `zfill`
* Add efficient (Numba-based) implementations for `endswith`, `startswith` and `contains`

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
 * Implemented `.str.cat` as `.fr_strx.cat` for arrays with `pa.string()` dtype.
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
