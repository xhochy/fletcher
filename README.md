# fletcher

[![CircleCI](https://circleci.com/gh/xhochy/fletcher/tree/master.svg?style=svg)](https://circleci.com/gh/xhochy/fletcher/tree/master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fxhochy%2Ffletcher.svg?type=shield)](https://app.fossa.io/projects/git%2Bgithub.com%2Fxhochy%2Ffletcher?ref=badge_shield)

A library that provides a generic set of Pandas ExtensionDType/Array
implementations backed by Apache Arrow. They support a wider range of types
than Pandas natively supports and also bring a different set of constraints and
behaviours that are beneficial in many situations.

## Usage

To use `fletcher` in Pandas DataFrames, all you need to do is to wrap your data
in a `FletcherChunkedArray` object. Your data can be of either `pyarrow.Array`,
`pyarrow.ChunkedArray` or a type that can be passed to `pyarrow.array(…)`.


```
import fletcher as fr
import pandas as pd

df = pd.DataFrame({
    'str': fr.FletcherChunkedArray(['a', 'b', 'c'])
})

df.info()

# RangeIndex: 3 entries, 0 to 2
# Data columns (total 1 columns):
# str    3 non-null fletcher[string]
# dtypes: fletcher[string](1)
# memory usage: 100.0 bytes
```

## Development

While you can use `fletcher` in pip-based environments, we strongly recommend
using a `conda` based development setup with packages from `conda-forge`.

```
# Create the conda environment with all necessary dependencies
conda create -y -q -n fletcher python=3.6 \
    pre-commit \
    asv \
    numba \
    pandas \
    pip \
    pyarrow \
    pytest \
    pytest-cov \
    six \
    -c conda-forge

# Activate the newly created environment
source activate fletcher

# Install fletcher into the current environment
pip install -e .

# Run the unit tests (you should do this several times during development)
py.test

# Install pre-commit hooks
# These will then be automatically run on every commit and ensure that files
# are black formatted, have no flake8 issues and mypy checks the type consistency.
pre-commit install
```

Code formatting is done using black. This should keep everything in a
consistent styling and the formatting can be automatically adjusted using
`black .`. Note that we have pinned the version of `black` to ensure that
the formatting is reproducible.

### Benchmarks

In `benchmarks/` we provide a set of benchmarks to compare the performance of
`fletcher` against `pandas` and ensure that `fletcher` itself stays performant.
The benchmarks are written using
[airspeed velocity](https://asv.readthedocs.io/en/stable/). When developing
the benchmarks you can run them using `asv dev` (use `-b <pattern>` to only
run a selection of them) only once. To get real benchmark values, you should
use `asv run --python=same` to run the benchmarks multiple times and get
meaningful average runtimes.
