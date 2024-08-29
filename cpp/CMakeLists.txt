if(WIN32)
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
  message("-- Detected conda environment, setting INSTALL_RPATH_USE_LINK_PATH")
  set(_using_conda On)
  set(_using_conda
      On
      PARENT_SCOPE)
else()
  set(_using_conda Off)
  set(_using_conda
      Off
      PARENT_SCOPE)
endif()

include(box/box-files.cmake)
include(locality/locality-files.cmake)
include(parallel/parallel-files.cmake)
include(pmft/pmft-files.cmake)
include(util/util-files.cmake)

# Use a static library for the main code. A Dynamic library would be more
# disk space efficient, but I was unable to resolve a RPATH issue where
# any use of `INSTALL_RPATH` would result in "terminated by signal SIGKILL"
# the moment Python attempts to load the library.
add_library(freud STATIC ${box_sources} ${locality_sources}
                            ${parallel_sources} ${pmft_sources} ${util_sources})
set_target_properties(freud PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(freud PUBLIC TBB::tbb)

target_include_directories(freud SYSTEM PUBLIC ${PROJECT_SOURCE_DIR}/extern/)

# now build the python extension modules
add_subdirectory(box)
add_subdirectory(locality)
add_subdirectory(parallel)
add_subdirectory(pmft)
add_subdirectory(util)

# commented out for now, uncomment them as the conversion to pybind11 progresses
# add_subdirectory(cluster) add_subdirectory(density)
# add_subdirectory(diffraction) add_subdirectory(environment)
# add_subdirectory(order) add_subdirectory(pmft)
