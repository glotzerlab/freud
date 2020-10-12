# We use the IN_LIST operator, which is first supported in 3.3.0. This is the
# oldest version of CMake that this build script has been tested with. However,
# the oldest CMake actively tested against will be determined by the oldest
# CMake available on PyPI (which is 3.6.3).
cmake_minimum_required(VERSION 3.3.0)

project(freud)

set(DEFAULT_BUILD_TYPE "Release")

# Set a default build type if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' since none was specified.")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
        STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# This setting appears to be required by older versions of CMake for object libraries.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/CMake")
find_package(TBB REQUIRED)

# Fail fast if users have not cloned submodules.
if(NOT WIN32)
  string(ASCII 27 Esc)
  set(Red         "${Esc}[31m")
  set(DefaultColor "${Esc}[m")
endif()
set(submodules
    Eigen
    fsph
    voro++)
foreach (submodule ${submodules})
    if( NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/${submodule}/.git" )
        message( FATAL_ERROR "${Red}The git submodules are not available. Please run git submodule update --init --recursive.${DefaultColor}"
        )
    endif()
endforeach(submodule)

include_directories(
    ${PROJECT_SOURCE_DIR}/cpp/util
    ${PROJECT_SOURCE_DIR}/cpp/locality
    ${PROJECT_SOURCE_DIR}/cpp/box
    ${PROJECT_SOURCE_DIR}/extern
    # For the order module
    ${PROJECT_SOURCE_DIR}/cpp/cluster
	${TBB_INCLUDE_DIR}
    )

add_subdirectory(cpp)
add_subdirectory(freud)
