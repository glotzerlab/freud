# CMake 3.12.0 is the oldest version supported by the way freud links TBB to
# object libraries like _cluster. This is also the oldest version tested in CI.
cmake_minimum_required(VERSION 3.12.0)

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

# This setting is required when building shared libraries from object libraries.
# CMake intentionally chooses not to infer such information, so we should expect
# to specify this.
# https://stackoverflow.com/questions/50600708/combining-cmake-object-libraries-with-shared-libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(CMake)
find_package_config_first(TBB)

if(TBB_FOUND)
  include(FindPackageMessage)
  find_package_message(
    tbb "Found TBB: ${TBB_DIR} ${TBB_LIBRARY} ${TBB_INCLUDE_DIR}"
    "[${TBB_LIBRARY}][${TBB_INCLUDE_DIR}]")
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

# We treat the extern folder as a SYSTEM library to avoid getting any diagnostic
# information from it. In particular, this avoids clang-tidy throwing errors due
# to any issues in external code. include_directories(SYSTEM ${TBB_INCLUDE_DIR})

# Ignore unused variable warning from scikit-build
set(ignoreMe "${SKBUILD}")

add_subdirectory(cpp)
add_subdirectory(freud)

if(_using_conda OR DEFINED ENV{CIBUILDWHEEL})
  set_target_properties(libfreud PROPERTIES INSTALL_RPATH_USE_LINK_PATH True)
endif()
