CMAKE_MINIMUM_REQUIRED(VERSION 2.6.2 FATAL_ERROR)

if (CMAKE_MAJOR_VERSION VERSION_GREATER 2)
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
    cmake_policy(SET CMP0026 OLD)
    cmake_policy(SET CMP0042 OLD)
endif(COMMAND cmake_policy)
endif()

project (FREUD)

add_subdirectory(CMake)

# find TBB lib and includes
find_library(TBB_LIBRARY tbb
             PATHS ENV TBB_LINK)
find_path(TBB_INCLUDE_DIR tbb/tbb.h
          PATHS ENV TBB_INC)
include_directories(${TBB_INCLUDE_DIR})
if (TBB_LIBRARY)
    mark_as_advanced(TBB_LIBRARY)
endif()
if (TBB_INCLUDE_DIR)
    mark_as_advanced(TBB_INCLUDE_DIR)
endif()

# find ISPC
# find_program(ISPC_EXECUTABLE ispc)
# if (ISPC_EXECUTABLE)
#     mark_as_advanced(ISPC_EXECUTABLE)
# endif()

include(FREUDCFlagsSetup)

# setup python
include(FREUDPythonSetup)
include_directories(${PYTHON_INCLUDE_DIR})

# setup cython
# include(FREUDCythonSetup)
find_package(Cython)
include(UseCython)

# Find the boost libraries and set them up
include (FREUDBoostSetup)
include_directories(SYSTEM ${Boost_INCLUDE_DIR})

# setup numpy
include (FREUDNumpySetup)
include_directories(${NUMPY_INCLUDE_DIR})
# cython uses the deprecated api, so these remain commented til further notice
# add_definitions(-DPY_ARRAY_UNIQUE_SYMBOL=PyArrayHandle)
# add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)

# macros
include (FREUDMacros)

# include external source projects
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/extern)

# set the default install prefix
IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    SET(CMAKE_INSTALL_PREFIX ${PYTHON_USER_SITE} CACHE PATH "Python site installation directory (defaults to USER_SITE)" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# Fixup conda linking, if this python appears to be a conda python
get_filename_component(_python_bin_dir ${PYTHON_EXECUTABLE} DIRECTORY)
if (EXISTS "${_python_bin_dir}/conda")
    message("-- Detected conda python, activating workaround")
    set(_using_conda On)
else()
    set(_using_conda Off)
endif()

#add_subdirectory(freud)
include(freud/CMakeLists.txt)
add_subdirectory(cpp)
# add_subdirectory(doc)
