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

# include(FREUDCFlagsSetup)

# setup python
include(FREUDPythonSetup)
# include_directories(${PYTHON_INCLUDE_DIR})

# Find the boost libraries and set them up
include (FREUDBoostSetup)
# include_directories(SYSTEM ${Boost_INCLUDE_DIR})

# setup numpy
# include (FREUDNumpySetup)
# include_directories(${NUMPY_INCLUDE_DIR})
# add_definitions(-DPY_ARRAY_UNIQUE_SYMBOL=PyArrayHandle)
# add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)

# set the default install prefix
IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    SET(SETUP_PY_ARGS "--user" CACHE STRING "Extra args to pass to setup.py")
ELSE(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    SET(SETUP_PY_ARGS "--prefix=\"${CMAKE_INSTALL_PREFIX}\"" CACHE STRING "Extra args to pass to setup.py")
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/freud/__init__.py")
set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")

configure_file(${SETUP_PY_IN} ${SETUP_PY})

add_custom_command(OUTPUT ${OUTPUT}
                   COMMAND "${PYTHON_EXECUTABLE}" setup.py build_ext
                   COMMAND "${CMAKE_COMMAND}" -E touch ${OUTPUT}
                   WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                   DEPENDS ${DEPS})

add_custom_target(target ALL DEPENDS ${OUTPUT})
install(CODE "execute_process(COMMAND \"${PYTHON_EXECUTABLE}\" setup.py install ${SETUP_PY_ARGS}
                              WORKING_DIRECTORY \"${CMAKE_CURRENT_BINARY_DIR}\")")

add_subdirectory(doc)
