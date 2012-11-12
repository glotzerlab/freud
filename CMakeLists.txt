CMAKE_MINIMUM_REQUIRED(VERSION 2.6.2 FATAL_ERROR)

project (FREUD)

add_subdirectory(CMake)

include(FREUDCFlagsSetup)

# setup boost
set(Boost_ADDITIONAL_VERSIONS "1.45.1;1.45.0;1.45;1.44.1;1.44.0;1.44;1.43.1;1.43.0;1.43;1.42.1;1.42.0;1.42;1.41.0;1.41;1.41;1.40.0;1.40;1.39.0;1.39;1.38.0;1.38")
find_package(Boost 1.35.0 COMPONENTS python REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIR})

# setup python
include(FREUDPythonSetup)
include_directories(${PYTHON_INCLUDE_DIR})

# setup numpy
include (FREUDNumpySetup)
include_directories(${NUMPY_INCLUDE_DIR})
add_definitions(-DPY_ARRAY_UNIQUE_SYMBOL=PyArrayHandle)

# setup OpenMP
# OpenMP needs CMake 2.6.4 or newer
set (_cmake_ver "${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}")
if (_cmake_ver VERSION_GREATER 2.6.3)
    find_package(OpenMP)
    if (OPENMP_FOUND)
        option (ENABLE_OPENMP "Enable openmp compilation to accelerate CPU code on multi-core machines" ON)
        if (ENABLE_OPENMP)
            add_definitions (-DENABLE_OPENMP)
            # Compiler options to enable openmp are set in FREUDCFlagsSetup
        endif (ENABLE_OPENMP)
    endif (OPENMP_FOUND)
else (_cmake_ver VERSION_GREATER 2.6.3)
    message (STATUS "Upgrade to CMake 2.6.4 or newer to enable OpenMP compilation")
endif (_cmake_ver VERSION_GREATER 2.6.3)

add_subdirectory(cpp)
add_subdirectory(py)
add_subdirectory(run)
add_subdirectory(doc)

