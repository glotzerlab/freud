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
TBB_VERSION="2022.1.0"
TBB_ZIP="v${TBB_VERSION}.zip"
curl -L -O "https://github.com/oneapi-src/oneTBB/archive/refs/tags/${TBB_ZIP}"
unzip -q "${TBB_ZIP}"

#
echo "PLATFORM=${PLATFORM}"
echo "ARCHFLAGS=${ARCHFLAGS}"
EXTRA_CMAKE_ARGS=""
if [[ "${PLATFORM}" == "macos" ]]; then
    EXTRA_CMAKE_ARGS="-DCMAKE_INSTALL_PREFIX=/Users/runner/work/tbb-install"
    if [[ ${ARCHFLAGS} == *"arm64"* ]]; then
        EXTRA_CMAKE_ARGS="-DCMAKE_OSX_ARCHITECTURES=arm64 ${EXTRA_CMAKE_ARGS}"
    elif [[ ${ARCHFLAGS} == *"x86_64"* ]]; then
        EXTRA_CMAKE_ARGS="-DCMAKE_OSX_ARCHITECTURES=x86_64 ${EXTRA_CMAKE_ARGS}"
    fi
fi
echo "EXTRA_CMAKE_ARGS=${EXTRA_CMAKE_ARGS}"

# clean the build to rebuild for arm64
rm -rf "${PACKAGE_DIR}/tbb"

# Move to a hard-coded path (defined by CIBW_ENVIRONMENT)
mv "oneTBB-${TBB_VERSION}" "${PACKAGE_DIR}/tbb"
cd "${PACKAGE_DIR}/tbb"
mkdir -p build
cd build
cmake ../ -DTBB_TEST=OFF -DTBB_STRICT=OFF -DCMAKE_BUILD_TYPE=Release ${EXTRA_CMAKE_ARGS}
cmake --build . -j -v
cmake --install .