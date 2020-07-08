# fletcher

![CI](https://github.com/xhochy/fletcher/workflows/CI/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/xhochy/fletcher/master)

A library that provides a generic set of Pandas ExtensionDType/Array
implementations backed by Apache Arrow. They support a wider range of types
than Pandas natively supports and also bring a different set of constraints and
behaviours that are beneficial in many situations.

## Usage

To use `fletcher` in Pandas DataFrames, all you need to do is to wrap your data
in a `FletcherChunkedArray` or `FletcherContinuousArray` object. Your data can 
be of either `pyarrow.Array`, `pyarrow.ChunkedArray` or a type that can be passed
to `pyarrow.array(…)`.


```
import fletcher as fr
import pandas as pd

df = pd.DataFrame({
    'str_chunked': fr.FletcherChunkedArray(['a', 'b', 'c']),
    'str_continuous': fr.FletcherContinuousArray(['a', 'b', 'c']),
})

df.info()

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 3 entries, 0 to 2
# Data columns (total 2 columns):
#  #   Column          Non-Null Count  Dtype                      
# ---  ------          --------------  -----                      
#  0   str_chunked     3 non-null      fletcher_chunked[string]   
#  1   str_continuous  3 non-null      fletcher_continuous[string]
# dtypes: fletcher_chunked[string](1), fletcher_continuous[string](1)
# memory usage: 166.0 bytes
```

## Development

While you can use `fletcher` in pip-based environments, we strongly recommend
using a `conda` based development setup with packages from `conda-forge`.

```
# Create the conda environment with all necessary dependencies
conda env create

# Activate the newly created environment
conda activate fletcher

# Install fletcher into the current environment
python -m pip install -e . --no-build-isolation --no-use-pep517

# Run the unit tests (you should do this several times during development)
py.test -nauto

# Install pre-commit hooks
# These will then be automatically run on every commit and ensure that files
# are black formatted, have no flake8 issues and mypy checks the type consistency.
pre-commit install
```

Code formatting is done using black. This should keep everything in a
consistent styling and the formatting is automatically adjusted via the
pre-commit hooks.

### Using pandas in development mode

To test and develop against pandas' master or your local fixes, you can install a development version of pandas using:

```
git clone https://github.com/pandas-dev/pandas
cd pandas

# Install additional pandas dependencies
conda install -y cython

# Build and install pandas
python setup.py build_ext --inplace -j 4
python -m pip install -e . --no-build-isolation --no-use-pep517
```

This links the development version of `pandas` into your `fletcher` conda environment.
If you change any Python code in pandas, it is directly reflected in your environment.
If you change any Cython code in pandas, you need to re-execute `python setup.py build_ext --inplace -j 4`.

### Using (py)arrow nightlies

To test and develop against the latest development version of Apache Arrow (`pyarrow`), you can install it from the `arrow-nightlies` conda channel:

```
conda install -c arrow-nightlies arrow-cpp pyarrow
```

### Benchmarks

In `benchmarks/` we provide a set of benchmarks to compare the performance of
`fletcher` against `pandas` and ensure that `fletcher` itself stays performant.
The benchmarks are written using
[airspeed velocity](https://asv.readthedocs.io/en/stable/). When developing
the benchmarks you can run them using `asv dev` (use `-b <pattern>` to only
run a selection of them) only once. To get real benchmark values, you should
use `asv run --python=same` to run the benchmarks multiple times and get
meaningful average runtimes.
