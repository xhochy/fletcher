# fletcher

[![CircleCI](https://circleci.com/gh/xhochy/fletcher/tree/master.svg?style=svg)](https://circleci.com/gh/xhochy/fletcher/tree/master)

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
    numba \
    pandas \
    pyarrow \
    pytest \
    pytest-cov \
    six \
    flake8 \
    pip \
    codecov \
    pip \
    -c conda-forge

# Activate the newly created environment
source activate fletcher

# Install fletcher into the current environment
pip install -e .

# Run the unit tests (you should do this several times during development)
py.test
```
