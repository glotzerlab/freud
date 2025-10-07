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

# copy all given files from the current source directory to the current build
# directory files must be specified by relative path
#
# @param files: list of files to copy @param target: name of copy target @param
# validate_pattern: Check ${CMAKE_CURRENT_BINARY_DIR}/${validate_pattern} for
# files that are not in ${files} and issue a warning. @param Additional
# parameters: List of files to ignore
function(copy_files_to_build files target validate_pattern)
  set(ignore_files ${ARGN})

  file(RELATIVE_PATH relative_dir ${PROJECT_BINARY_DIR}
       ${CMAKE_CURRENT_BINARY_DIR})

  foreach(file ${files})
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${file}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${file}
      COMMAND ${CMAKE_COMMAND} ARGS -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file}
              ${CMAKE_CURRENT_BINARY_DIR}/${file}
      COMMENT "Copy ${relative_dir}/${file}")
  endforeach()

  add_custom_target(copy_${target} ALL DEPENDS ${files})

  file(GLOB _matching_files "${CMAKE_CURRENT_BINARY_DIR}/${validate_pattern}")
  foreach(file ${_matching_files})
    # message("Matching files: ${_matching_files}")
    file(RELATIVE_PATH relative_file ${CMAKE_CURRENT_BINARY_DIR} ${file})
    # message("Expected files: ${files}")
    list(FIND files ${relative_file} found)
    if(found EQUAL -1 AND NOT ${relative_file} IN_LIST ignore_files)
      message(WARNING "${file} exists but is not provided by the source. "
                      "Remove this file to prevent unexpected errors.")
    endif()
  endforeach()
endfunction()
