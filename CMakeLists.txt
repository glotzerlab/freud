CMAKE_MINIMUM_REQUIRED(VERSION 2.6.2 FATAL_ERROR)

project (FREUD)

add_subdirectory(CMake)

find_library(TBB_LIBRARY tbb)
find_program(ISPC_EXECUTABLE ispc)

include(FREUDCFlagsSetup)

# setup python
include(FREUDPythonSetup)
include_directories(${PYTHON_INCLUDE_DIR})

# Find the boost libraries and set them up
include (FREUDBoostSetup)
include_directories(SYSTEM ${Boost_INCLUDE_DIR})

# setup numpy
include (FREUDNumpySetup)
include_directories(${NUMPY_INCLUDE_DIR})
add_definitions(-DPY_ARRAY_UNIQUE_SYMBOL=PyArrayHandle)


add_subdirectory(cpp)
add_subdirectory(py)
add_subdirectory(run)
add_subdirectory(doc)

