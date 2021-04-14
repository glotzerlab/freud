#!/bin/bash
if [[ $(python --version 2>&1) == *"3.6."* ]]; then
  # Python 3.6 is only supported with oldest requirements
  pip install -U -r /project/.circleci/ci-oldest-reqs.txt --progress-bar=off
else
  pip install -U -r /project/requirements/requirements-test.txt --progress-bar=off
fi
