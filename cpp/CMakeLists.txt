if(WIN32)
  # Use add_compile_definitions when dropping support for CMake < 3.12 Force
  # Windows to define M_PI in <cmath>
  add_compile_options(/D_USE_MATH_DEFINES)
  # Prevent Windows from defining min/max as macros
  add_compile_options(/DNOMINMAX)
endif()

# Detect when building against a conda environment set the _using_conda variable
# for use both in this file and in the parent
get_filename_component(_python_bin_dir ${PYTHON_EXECUTABLE} DIRECTORY)
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

add_subdirectory(box)

# commented out for now, uncomment them as the conversion to pybind11 progresses
# add_subdirectory(cluster) add_subdirectory(density)
# add_subdirectory(diffraction) add_subdirectory(environment)
# add_subdirectory(locality) add_subdirectory(order) add_subdirectory(parallel)
# add_subdirectory(pmft) add_subdirectory(util)

#[[
add_library(
  libfreud SHARED
  $<TARGET_OBJECTS:_cluster>
  $<TARGET_OBJECTS:_density>
  $<TARGET_OBJECTS:_diffraction>
  $<TARGET_OBJECTS:_environment>
  $<TARGET_OBJECTS:_locality>
  $<TARGET_OBJECTS:_order>
  $<TARGET_OBJECTS:_parallel>
  $<TARGET_OBJECTS:_pmft>
  $<TARGET_OBJECTS:_util>)

target_link_libraries(libfreud PUBLIC TBB::tbb)

# We treat the extern folder as a SYSTEM library to avoid getting any diagnostic
# information from it. In particular, this avoids clang-tidy throwing errors due
# to any issues in external code.
target_include_directories(libfreud SYSTEM PUBLIC ${PROJECT_SOURCE_DIR}/extern/)

# CMake will automatically prepend any library name with "lib" when creating
# shared libraries, so a CMake library named libfreud will result in
# liblibfreud.so object. This rename gives the more expected name of
# libfreud.so. Note that we choose not to name the CMake target "freud" to avoid
# conflicting with the project name.
set_target_properties(libfreud PROPERTIES OUTPUT_NAME freud)

# Copy the C++ library into the built version.
install(TARGETS libfreud DESTINATION freud)

if(CMAKE_EXPORT_COMPILE_COMMANDS)
  # Copy the compile commands into the root of the project.
  add_custom_command(
    TARGET libfreud
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/compile_commands.json
            ${PROJECT_SOURCE_DIR})
endif()
]]
