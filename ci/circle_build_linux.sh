#!/bin/bash

set +x
set -eo pipefail

export PYTHON_VERSION=$1
export USE_DEV_WHEELS=$2
export CONDA_PKGS_DIRS=$HOME/.conda_packages
export MINICONDA=$HOME/miniconda
export MINICONDA_URL="https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh"
export PATH="$MINICONDA/bin:$PATH"

wget --no-verbose -O miniconda.sh $MINICONDA_URL
bash miniconda.sh -b -p $MINICONDA
export PATH="$MINICONDA/bin:$PATH"

conda update -y -q conda
conda config --set auto_update_conda false
conda config --add channels https://repo.continuum.io/pkgs/free
conda config --add channels conda-forge

conda create -y -q -n fletcher python=${PYTHON_VERSION} \
    pandas pyarrow pytest pytest-cov pytest-flake8 \
    flake8 \
    setuptools_scm \
    pip \
    numba \
    codecov \
    six \
    sphinx \
    -c conda-forge
source activate fletcher

if [ "${PYTHON_VERSION}" = "3.6" ]; then
  conda install -y -q black=18.5b0 -c conda-forge
  black --check .
fi

if [[ ${USE_DEV_WHEELS} ]]; then
    echo "Installing NumPy and Pandas dev"
    conda uninstall -y --force numpy pandas
    PRE_WHEELS="https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com"
    pip install --pre --no-deps --upgrade --timeout=60 -f $PRE_WHEELS numpy pandas
fi

pip install -e .
py.test --junitxml=test-reports/junit.xml --cov=./

# Do a second run with JIT disabled to produce coverage and check that the
# code works also as expected in Python.
if [ "${PYTHON_VERSION}" = "3.6" ]; then
  # These don't work with Python 2.7 as it supports less operators than 3.6
  NUMBA_DISABLE_JIT=1 py.test --junitxml=test-reports/junit.xml --cov=./
fi
# Upload coverage in each build, codecov.io merges the reports
codecov

# Check documentation build only in one job
if [ "${PYTHON_VERSION}" = "3.6" ]; then
  pushd docs
  make html
  popd
fi
