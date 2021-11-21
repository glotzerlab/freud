#!/bin/bash
if [ -z $1 ]; then
    echo "A package directory must be provided as the first argument."
    exit 1
fi
if [ -z $2 ]; then
    echo "A platform {linux,macos} must be provided as the second argument."
    exit 1
fi

# Install a modern version of CMake for compatibility with TBB 2021
# (manylinux image includes CMake 2.8)
pip install cmake

PACKAGE_DIR=$1
PLATFORM=$2
TBB_VERSION="2021.2.0"
TBB_ZIP="v${TBB_VERSION}.zip"
curl -L -O "https://github.com/oneapi-src/oneTBB/archive/refs/tags/${TBB_ZIP}"
unzip -q "${TBB_ZIP}"

# Move to a hard-coded path (defined by CIBW_ENVIRONMENT)
mv "oneTBB-${TBB_VERSION}" "${PACKAGE_DIR}/tbb"
cd "${PACKAGE_DIR}/tbb"
mkdir -p build
cd build
cmake ../ -DTBB_TEST=OFF -DTBB_STRICT=OFF
cmake --build . -j
cmake --install .
