cmake_minimum_required(VERSION 3.10)

project(freud)

set(DEFAULT_BUILD_TYPE "Release")

# Set a default build type if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' since none was specified.")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
        STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/CMake")
find_package(TBB REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/cpp/util
    ${PROJECT_SOURCE_DIR}/cpp/locality
    ${PROJECT_SOURCE_DIR}/cpp/box
	${TBB_INCLUDE_DIR}
    )

add_subdirectory(cpp)
add_subdirectory(freud)
