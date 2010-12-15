CMAKE_MINIMUM_REQUIRED(VERSION 2.6.2 FATAL_ERROR)

project (FREUD)
add_subdirectory(CMake)

# setup boost
set(Boost_ADDITIONAL_VERSIONS "1.45.1;1.45.0;1.45;1.44.1;1.44.0;1.44;1.43.1;1.43.0;1.43;1.42.1;1.42.0;1.42;1.41.0;1.41;1.41;1.40.0;1.40;1.39.0;1.39;1.38.0;1.38")
find_package(Boost 1.35.0 COMPONENTS python REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIR})

# setup python
find_package(PythonInterp REQUIRED)
set(CMAKE_FIND_FRAMEWORK "LAST")
find_package(PythonLibs REQUIRED)

include_directories(${PYTHON_INCLUDE_DIR})

# setup numpy
include (FREUDNumpySetup)
include_directories(${NUMPY_INCLUDE_DIR})
add_definitions(-DPY_ARRAY_UNIQUE_SYMBOL=PyArrayHandle)

add_subdirectory(cpp)
add_subdirectory(py)
add_subdirectory(run)

