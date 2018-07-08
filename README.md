# fletcher

[![CircleCI](https://circleci.com/gh/xhochy/fletcher/tree/master.svg?style=svg)](https://circleci.com/gh/xhochy/fletcher/tree/master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

A library that provides a generic set of Pandas ExtensionDType/Array
implementations backed by Apache Arrow. They support a wider range of types
than Pandas natively supports and also bring a different set of constraints and
behaviours that are beneficial in many situations.

## Usage

To use `fletcher` in Pandas DataFrames, all you need to do is to wrap your data
in a `FletcherArray` object. Your data can be of either `pyarrow.Array`,
`pyarrow.ChunkedArray` or a type that can be passed to `pyarrow.array(…)`.


```
import fletcher as fr
import pandas as pd

df = pd.DataFrame({
    'str': fr.FletcherArray(['a', 'b', 'c'])
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
    black=18.5b0 \
    codecov \
    flake8 \
    numba \
    pandas \
    pip \
    pyarrow \
    pytest \
    pytest-cov \
    pytest-flake8 \
    six \
    -c conda-forge

# Activate the newly created environment
source activate fletcher

# Install fletcher into the current environment
pip install -e .

# Run the unit tests (you should do this several times during development)
py.test
```

Code formatting is done using black. This should keep everything in a
consistent styling and the formatting can be automatically adjusted using
`black .`. Note that we have pinned the version of `black` to ensure that
the formatting is reproducible.
