# fletcher

[![CircleCI](https://circleci.com/gh/xhochy/fletcher/tree/master.svg?style=svg)](https://circleci.com/gh/xhochy/fletcher/tree/master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

A library that provides a generic set of Pandas ExtensionDType/Array
implementations backed by Apache Arrow. They support a wider range of types
than Pandas natively supports and also bring a different set of constraints and
behaviours that are beneficial in many situations.

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
