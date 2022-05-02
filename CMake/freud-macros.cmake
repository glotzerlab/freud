# Use CMake's newer config-based package finding before falling back to the old
# find package mechanism.
macro(find_package_config_first package)
  # VERSION_GREATER_EQUAL requires CMake >=3.7, so use NOT ... VERSION_LESS
  if(NOT ${CMAKE_VERSION} VERSION_LESS 3.15)
    set(_old_prefer_config ${CMAKE_FIND_PACKAGE_PREFER_CONFIG})
    set(CMAKE_FIND_PACKAGE_PREFER_CONFIG TRUE)
    find_package(${package} ${ARGN})
    set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ${_old_prefer_config})
  else()
    find_package(${package} QUIET CONFIG ${ARGN})
    if(NOT ${package}_FOUND)
      find_package(${package} MODULE ${ARGN} REQUIRED)
    endif()
  endif()
endmacro()

# Convert shared libraries to python extensions
function(make_python_extension_module _target)
  set_target_properties(${_target} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}")
  set_target_properties(${_target}
                        PROPERTIES SUFFIX "${PYTHON_EXTENSION_MODULE_SUFFIX}")

  target_include_directories(${_target} PRIVATE "${PYTHON_INCLUDE_DIRS}")
  if(WIN32)
    # Link to the Python libraries on windows
    target_link_libraries(${_target} ${PYTHON_LIBRARIES})
  else()
    # Do not link to the Python libraries on Mac/Linux - symbols are provided by
    # the `python` executable. "-undefined dynamic_lookup" is needed on Mac
    target_link_options(
      ${_target} PRIVATE
      "$<$<PLATFORM_ID:Darwin>:LINKER:-undefined,dynamic_lookup>")
  endif()
endfunction()
