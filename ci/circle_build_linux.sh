#!/bin/bash

set +x
set -eo pipefail

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTHON_VERSION=$1
export USE_DEV_WHEELS=$2
export CONDA_PKGS_DIRS=$HOME/.conda_packages
export MINICONDA=$HOME/miniconda
export MINICONDA_URL="https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh"
export PATH="$MINICONDA/bin:$PATH"

wget --no-verbose -O miniconda.sh $MINICONDA_URL
bash miniconda.sh -b -p $MINICONDA
export PATH="$MINICONDA/bin:$PATH"

conda config --set auto_update_conda false
conda config --add channels https://repo.continuum.io/pkgs/free
conda config --add channels conda-forge
conda install -y mamba

if [ "${USE_DEV_WHEELS}" = "nightlies" ]; then
    export CONDA_ARROW="arrow-nightlies::pyarrow arrow-nightlies::arrow-cpp -c arrow-nightlies"
else
    export CONDA_ARROW="pyarrow arrow-cpp"
fi

mamba create -y -q -n fletcher python=${PYTHON_VERSION} \
    'pandas>=1' pytest pytest-cov \
    hypothesis \
    setuptools_scm \
    pip \
    'numba>=0.49' \
    codecov \
    six \
    sphinx \
    sphinx_rtd_theme \
    numpydoc \
    sphinxcontrib-apidoc \
    pre_commit \
    dask \
    $CONDA_ARROW \
    -c conda-forge
source activate fletcher

if [ "${PYTHON_VERSION}" = "3.7" ]; then
  pre-commit install
  pre-commit run -a
fi

if [ "${USE_DEV_WHEELS}" = "nightlies" ]; then
    echo "Installing Pandas dev"
    conda uninstall -y --force pandas
    PRE_WHEELS="https://pypi.anaconda.org/scipy-wheels-nightly/simple"
    pip install --pre --no-deps --upgrade --timeout=60 -i $PRE_WHEELS pandas
fi

pip install --no-deps -e .
py.test --junitxml=test-reports/junit.xml --cov=./ --cov-report=xml

# Do a second run with JIT disabled to produce coverage and check that the
# code works also as expected in Python.
if [ "${PYTHON_VERSION}" = "3.6" ]; then
  # These don't work with Python 2.7 as it supports less operators than 3.6
  NUMBA_DISABLE_JIT=1 py.test --junitxml=test-reports/junit.xml --cov=./ --cov-report=xml
fi

# Check documentation build only in one job
if [ "${PYTHON_VERSION}" = "3.7" ]; then
  pushd docs
  make html
  popd
fi
