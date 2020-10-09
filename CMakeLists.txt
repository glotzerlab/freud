cmake_minimum_required(VERSION 3.10)

# set the project name
project(freud)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    /Users/vyasramasubramani/local/freud/cpp/util
    /Users/vyasramasubramani/local/freud/cpp/locality
    /Users/vyasramasubramani/local/freud/cpp/box
	/Users/vyasramasubramani/local/freud/cpp/parallel
    /Users/vyasramasubramani/local/freud/extern
    ${NumPy_INCLUDE_DIRS}
    )

add_subdirectory(cpp)
