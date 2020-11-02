# We use the IN_LIST operator, which is first supported in 3.3.0. This is the
# oldest version of CMake that this build script has been tested with. However,
# the oldest CMake actively tested against will be determined by the oldest
# CMake available on PyPI (which is 3.6.3).
cmake_minimum_required(VERSION 3.3.0)

project(freud)

set(DEFAULT_BUILD_TYPE "Release")

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(
    STATUS
      "Setting build type to '${DEFAULT_BUILD_TYPE}' since none was specified.")
  set(CMAKE_BUILD_TYPE
      "${DEFAULT_BUILD_TYPE}"
      CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
                                               "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# This setting is required when building shared libraries from object libraries.
# CMake intentionally chooses not to infer such information, so we should expect
# to specify this.
# https://stackoverflow.com/questions/50600708/combining-cmake-object-libraries-with-shared-libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(CMake)
find_package_config_first(TBB)

# The TBB config doesn't define these variable, which we use later.
if(NOT DEFINED TBB_INCLUDE_DIR OR NOT DEFINED TBB_LIBRARY)
  get_target_property(TBB_INCLUDE_DIR TBB::tbb INTERFACE_INCLUDE_DIRECTORIES)
  get_target_property(TBB_LIBRARY TBB::tbb IMPORTED_LOCATION_RELEASE)
  set(TBB_USING_CONFIG "TRUE")
else()
  set(TBB_USING_CONFIG "FALSE")
endif()

# Fail fast if users have not cloned submodules.
if(NOT WIN32)
  string(ASCII 27 Esc)
  set(Red "${Esc}[31m")
  set(DefaultColor "${Esc}[m")
endif()
set(submodules Eigen fsph voro++)
foreach(submodule ${submodules})
  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/${submodule}/.git")
    message(
      FATAL_ERROR
        "${Red}Not all git submodules are available. Please run git submodule update --init --recursive.${DefaultColor}"
    )
  endif()
endforeach()

# Define preprocessor directives for Windows
if(WIN32)
  # Export all symbols (forces creation of .def file)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  # Use add_compile_definitions when dropping support for CMake < 3.12 Force
  # Windows to define M_PI in <cmath>
  add_compile_options(/D_USE_MATH_DEFINES)
  # Prevent Windows from defining min/max as macros
  add_compile_options(/DNOMINMAX)
endif()

include_directories(
  ${PROJECT_SOURCE_DIR}/cpp/util ${PROJECT_SOURCE_DIR}/cpp/locality
  ${PROJECT_SOURCE_DIR}/cpp/box ${TBB_INCLUDE_DIR})

# Ignore unused variable warning from scikit-build
set(ignoreMe "${SKBUILD}")

add_subdirectory(cpp)
add_subdirectory(freud)
