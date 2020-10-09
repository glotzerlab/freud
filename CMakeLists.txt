cmake_minimum_required(VERSION 3.10)

# set the project name
project(freud)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
find_package(TBB REQUIRED)

include_directories(
    /Users/vyasramasubramani/local/freud/cpp/util
    /Users/vyasramasubramani/local/freud/cpp/locality
    /Users/vyasramasubramani/local/freud/cpp/box
	/Users/vyasramasubramani/local/freud/cpp/parallel
    /Users/vyasramasubramani/local/freud/extern
	${TBB_INCLUDE_DIR}
    )

# add_library(TBB SHARED IMPORTED)
#set_target_properties(tbb PROPERTIES
#    IMPORTED_LOCATION "${TBB_LIBRARY}"
#    INTERFACE_INCLUDE_DIRECTORIES "${TBB_INCLUDE_DIR}"
#)

# I'm not 100% sure why I need to add this as well. It seems like the
# target_link_libraries below only adds the -ltbb, but it does not append
# to the library path. Need to figure out the most idiomatic way to do that.
get_filename_component(TBB_LIBRARY_DIR ${TBB_LIBRARY} DIRECTORY)
link_directories(${TBB_LIBRARY_DIR})

add_subdirectory(cpp)
add_subdirectory(freud)
