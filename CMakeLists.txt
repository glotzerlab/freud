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

# shared nanobind linking seems broken on windows for unknown reasons
if(WIN32)
  set(freud_nanobind_linking "NB_STATIC")
else()
  set(freud_nanobind_linking "NB_SHARED")
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
  ${PROJECT_SOURCE_DIR}/cpp/box)

add_subdirectory(cpp)
add_subdirectory(freud)
