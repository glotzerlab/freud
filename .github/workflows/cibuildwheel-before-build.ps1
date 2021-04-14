$PackageDir = $args[0]

# Install a modern version of CMake for compatibility with TBB 2021
# (manylinux image includes CMake 2.8)
#pip install cmake

$TBB_VERSION = "2021.2.0"
$TBB_ZIP = "v${TBB_VERSION}.zip"
curl -L -O "https://github.com/oneapi-src/oneTBB/archive/refs/tags/${TBB_ZIP}"
unzip -q "${TBB_ZIP}"

# Move to a hard-coded path (defined by CIBW_ENVIRONMENT)
mv "oneTBB-${TBB_VERSION}" "${PACKAGE_DIR}/tbb"
cd "${PACKAGE_DIR}/tbb"
mkdir -p build
cd build
cmake ../ -DTBB_TEST=OFF
make
cmake -DCOMPONENT=runtime -P cmake_install.cmake
cmake -DCOMPONENT=devel -P cmake_install.cmake
