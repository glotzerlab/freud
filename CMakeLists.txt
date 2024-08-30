cmake_minimum_required(VERSION 3.15...3.30)

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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(CMake)

find_package(
  Python 3.9
  COMPONENTS Interpreter Development.Module
  REQUIRED)

find_package_config_first(TBB)

if(TBB_FOUND)
  include(FindPackageMessage)
  find_package_message(
    tbb "Found TBB: ${TBB_DIR} ${TBB_LIBRARY} ${TBB_INCLUDE_DIR}"
    "[${TBB_LIBRARY}][${TBB_INCLUDE_DIR}]")
endif()

# go find nanobind
execute_process(
  COMMAND ${Python_EXECUTABLE} "-m" "nanobind" "--cmake_dir"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED 2.0)
if(nanobind_FOUND)
  find_package_message(
    nanobind "Found nanobind: ${nanobind_DIR} ${nanobind_VERSION}"
    "[${nanobind_DIR},${nanobind_VERSION}]")
endif()

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

# Detect when building against a conda environment set the _using_conda variable
# for use both in this file and in the parent
get_filename_component(_python_bin_dir ${Python_EXECUTABLE} DIRECTORY)
if(EXISTS "${_python_bin_dir}/../conda-meta")
  message(STATUS "Detected conda environment, setting INSTALL_RPATH_USE_LINK_PATH")
  set(_using_conda On)
else()
  set(_using_conda Off)
endif()

include_directories(
  ${PROJECT_SOURCE_DIR}/freud/box
  ${PROJECT_SOURCE_DIR}/freud/cluster
  ${PROJECT_SOURCE_DIR}/freud/density
  ${PROJECT_SOURCE_DIR}/freud/diffraction
  ${PROJECT_SOURCE_DIR}/freud/environment
  ${PROJECT_SOURCE_DIR}/freud/locality
  ${PROJECT_SOURCE_DIR}/freud/order
  ${PROJECT_SOURCE_DIR}/freud/parallel
  ${PROJECT_SOURCE_DIR}/freud/pmft
  ${PROJECT_SOURCE_DIR}/freud/util
  )

add_subdirectory(freud)
