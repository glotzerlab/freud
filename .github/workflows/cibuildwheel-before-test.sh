#!/bin/bash
if [ -z $1 ]; then
    echo "A package directory must be provided as the first argument."
    exit 1
fi

PACKAGE_DIR=$1
pip install -U -r "${PACKAGE_DIR}/requirements/requirements-test.txt" --progress-bar=off

# Allow parallel tests to speed up CI
pip install -U pytest-xdist --progress-bar=off