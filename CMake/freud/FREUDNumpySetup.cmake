# sets the variable NUMPY_INCLUDE_DIR for using numpy with python
find_package(PythonInterp REQUIRED)

if(NOT NUMPY_INCLUDE_DIR)

execute_process(
    COMMAND
    ${PYTHON_EXECUTABLE} -c "from __future__ import print_function; import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_GUESS
    RESULT_VARIABLE NUMPY_ERR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )

if(NUMPY_ERR)
    message(STATUS "Error while querying numpy include directory")
endif(NUMPY_ERR)

# We use the full path name (including numpy on the end), but
# Double-check that all is well with that choice.
find_path(
    NUMPY_INCLUDE_DIR
    numpy/arrayobject.h
    HINTS
    ${NUMPY_INCLUDE_GUESS}
    )

if (NUMPY_INCLUDE_DIR)
message(STATUS "Found numpy: ${NUMPY_INCLUDE_DIR}")
endif (NUMPY_INCLUDE_DIR)

endif(NOT NUMPY_INCLUDE_DIR)

if (NUMPY_INCLUDE_DIR)
mark_as_advanced(NUMPY_INCLUDE_DIR)
endif (NUMPY_INCLUDE_DIR)
