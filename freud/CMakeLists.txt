# Try alternate method for creating python extensions
function(my_python_extension_module _target)
  set(one_ops LINKED_MODULES_VAR FORWARD_DECL_MODULES_VAR MODULE_SUFFIX)
  cmake_parse_arguments(_args "" "${one_ops}" "" ${ARGN})

  set(_lib_type "NA")
  if(TARGET ${_target})
    get_property(_lib_type TARGET ${_target} PROPERTY TYPE)
  endif()

  set(_is_non_lib TRUE)

  set(_is_static_lib FALSE)
  if(_lib_type STREQUAL "STATIC_LIBRARY")
      set(_is_static_lib TRUE)
      set(_is_non_lib FALSE)
  endif()

  set(_is_shared_lib FALSE)
  if(_lib_type STREQUAL "SHARED_LIBRARY")
      set(_is_shared_lib TRUE)
      set(_is_non_lib FALSE)
  endif()

  set(_is_module_lib FALSE)
  if(_lib_type STREQUAL "MODULE_LIBRARY")
      set(_is_module_lib TRUE)
      set(_is_non_lib FALSE)
  endif()

  if(_is_static_lib OR _is_shared_lib OR _is_non_lib)

    if(_is_static_lib OR _is_shared_lib)
      if(_args_LINKED_MODULES_VAR)
        set(${_args_LINKED_MODULES_VAR}
            ${${_args_LINKED_MODULES_VAR}} ${_target} PARENT_SCOPE)
      else()
        set_property(GLOBAL APPEND PROPERTY PY_LINKED_MODULES_LIST ${_target})
      endif()
    endif()

    if(_args_FORWARD_DECL_MODULES_VAR)
      set(${_args_FORWARD_DECL_MODULES_VAR}
          ${${_args_FORWARD_DECL_MODULES_VAR}} ${_target} PARENT_SCOPE)
    else()
      set_property(GLOBAL APPEND PROPERTY
                   PY_FORWARD_DECL_MODULES_LIST ${_target})
    endif()
  endif()

  if(NOT _is_non_lib)
    include_directories("${PYTHON_INCLUDE_DIRS}")
  endif()

  if(_is_module_lib)
    set_target_properties(${_target} PROPERTIES
                          PREFIX "${PYTHON_MODULE_PREFIX}")
  endif()

  if(_is_module_lib OR _is_shared_lib)
    if(_is_module_lib)

      if(NOT _args_MODULE_SUFFIX)
        set(_args_MODULE_SUFFIX "${PYTHON_EXTENSION_MODULE_SUFFIX}")
      endif()

      if(_args_MODULE_SUFFIX STREQUAL "" AND WIN32 AND NOT CYGWIN)
        set(_args_MODULE_SUFFIX ".pyd")
      endif()

      if(NOT _args_MODULE_SUFFIX STREQUAL "")
        set_target_properties(${_target}
          PROPERTIES SUFFIX ${_args_MODULE_SUFFIX})
      endif()
    endif()

    #target_link_libraries_with_dynamic_lookup(${_target} ${PYTHON_LIBRARIES})
    target_link_options(${_target} PRIVATE -undefined dynamic_lookup)

    if(_is_module_lib)
      _set_python_extension_symbol_visibility(${_target})
    endif()
  endif()
endfunction()

# Need to figure out coverage before CYTHON_FLAGS are set.
option(COVERAGE "Enable coverage" OFF)

# Cython flags must be set before we run find_package for Cython since the
# compiler command is created immediately.
set(CYTHON_FLAGS
    "--directive binding=True,boundscheck=False,wraparound=False,embedsignature=True,always_allow_keywords=True"
    CACHE STRING "The directives for Cython compilation.")

if(COVERAGE)
  set(CYTHON_FLAGS
      "${CYTHON_FLAGS},linetrace=True"
      CACHE STRING "The directives for Cython compilation." FORCE)
endif()

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)

include_directories(${NumPy_INCLUDE_DIRS})

# Avoid Cython/Python3.8 minor incompatibility warnings, see
# https://github.com/cython/cython/issues/3474. Note that this option is a bit
# expansive, but it's a temporary fix and we'll be testing on other Python
# versions concurrently so it shouldn't hide any real issues. For backwards
# compatibility with older CMake, I'm using PythonInterp; when we drop support
# for CMake < 3.12, we should switch to find_package(Python).
find_package(PythonInterp REQUIRED)
if(${PYTHON_VERSION_MAJOR} EQUAL 3
   AND ${PYTHON_VERSION_MINOR} EQUAL 8
   AND NOT WIN32)
  add_compile_options("-Wno-deprecated-declarations")
endif()

# Detect when building against a conda environment set the _using_conda variable
# for use both in this file and in the parent
get_filename_component(_python_bin_dir ${PYTHON_EXECUTABLE} DIRECTORY)
#if(EXISTS "${_python_bin_dir}/../conda-meta")
#  message("-- Detected conda environment, setting INSTALL_RPATH_USE_LINK_PATH")
#  set(_using_conda On)
#  set(_using_conda
#      On
#      PARENT_SCOPE)
#else()
  set(_using_conda Off)
  set(_using_conda
      Off
      PARENT_SCOPE)
#endif()

set(cython_modules_with_cpp
    box
    cluster
    density
    diffraction
    environment
    locality
    order
    parallel
    pmft)

set(cython_modules_without_cpp interface msd util)

foreach(cython_module ${cython_modules_with_cpp} ${cython_modules_without_cpp})
  add_cython_target(${cython_module} PY3 CXX)
  add_library(${cython_module} MODULE ${${cython_module}})
  # For some reason this command prints the name of the resulting module. Would
  # be good to suppress if possible. (this is done by scikit build in
  # _set_python_extension_symbol_visibility)
  my_python_extension_module(${cython_module})

  target_compile_definitions(
    ${cython_module}
    # Avoid deprecation warnings for unsupported NumPy API versions. See
    # https://numpy.org/doc/1.19/reference/c-api/deprecations.html
    PRIVATE "NPY_NO_DEPRECATED_API=NPY_1_10_API_VERSION"
    # Default voro++ verbosity is high.
    PRIVATE "VOROPP_VERBOSE=1")
  if(COVERAGE)
    target_compile_definitions(
      ${cython_module} # Enable line tracing for coverage purposes if requested.
      PRIVATE "CYTHON_TRACE_NOGIL=1")
  endif()

  target_link_libraries(${cython_module} libfreud)

  # Separate logic required for targets with C++ code.
  if("${cython_module}" IN_LIST cython_modules_with_cpp)

    target_include_directories(
      ${cython_module} PRIVATE ${PROJECT_SOURCE_DIR}/cpp/${cython_module})
  endif()

  install(TARGETS ${cython_module} DESTINATION freud)

  # Coverage requires the Cython-compiled C++ files for line coverage.
  if(COVERAGE)
    install(FILES ${${cython_module}} DESTINATION freud)
  endif()

  if(APPLE)
    set_target_properties(${cython_module} PROPERTIES INSTALL_RPATH
                                                      "@loader_path")
  else()
    set_target_properties(${cython_module} PROPERTIES INSTALL_RPATH "\$ORIGIN")
  endif()

  if(_using_conda OR DEFINED ENV{CIBUILDWHEEL})
    set_target_properties(${cython_module}
                          PROPERTIES INSTALL_RPATH_USE_LINK_PATH True)
  endif()
endforeach()

# The SolidLiquid class has an instance of cluster::Cluster as a member, so
# including the header requires the Cluster.h header. Would prefer to inherit
# this information from the _order library, but that's not possible since we're
# linking to libfreud.
target_include_directories(order PUBLIC ${PROJECT_SOURCE_DIR}/cpp/cluster)
