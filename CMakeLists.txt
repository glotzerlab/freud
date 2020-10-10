cmake_minimum_required(VERSION 3.10)

project(freud)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/CMake")
find_package(TBB REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/cpp/util
    ${PROJECT_SOURCE_DIR}/cpp/locality
    ${PROJECT_SOURCE_DIR}/cpp/box
    ${PROJECT_SOURCE_DIR}/extern
    ${PROJECT_SOURCE_DIR}/extern/voro++/src
	${TBB_INCLUDE_DIR}
    )

add_subdirectory(extern/voro++)
add_subdirectory(cpp)
add_subdirectory(freud)
