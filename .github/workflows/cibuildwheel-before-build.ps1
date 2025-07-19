$PACKAGE_DIR = $args[0]

# Install a modern version of CMake for compatibility with modern Visual Studio
pip install "cmake==4.0.3"

$TBB_VERSION = "2022.2.0"
$TBB_ZIP = "v${TBB_VERSION}.zip"
Invoke-WebRequest -Uri "https://github.com/oneapi-src/oneTBB/archive/refs/tags/${TBB_ZIP}" -OutFile "${TBB_ZIP}"
Expand-Archive -Path "${TBB_ZIP}" -DestinationPath .

# Move to a hard-coded path (defined by CIBW_ENVIRONMENT)
mv "oneTBB-${TBB_VERSION}" "${PACKAGE_DIR}/tbb"
cd "${PACKAGE_DIR}/tbb"
mkdir -p build
cd build
cmake ../ -DTBB_TEST=OFF -DTBB_STRICT=OFF
cmake --build . -j --config Release
cmake --install . --config Release
