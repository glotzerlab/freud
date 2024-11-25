#!/bin/bash

# Execute this script to update all lock files to the latest versions of dependencies.

rm requirements*.txt

uv pip compile --python-version 3.13 --python-platform linux requirements-build-sdist.in > requirements-build-sdist.txt
