#!/bin/bash
if [ -z $1 ]; then
    echo "A package directory must be provided as the first argument."
    exit 1
fi

PACKAGE_DIR=$1

if [[ $(python --version 2>&1) == *"3.6."* ]]; then
  # Python 3.6 is only supported with oldest requirements
  pip install -U -r "${PACKAGE_DIR}/.circleci/ci-oldest-reqs.txt" --progress-bar=off
else
  pip install -U -r "${PACKAGE_DIR}/requirements/requirements-test.txt" --progress-bar=off
fi
