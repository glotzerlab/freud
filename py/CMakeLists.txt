if (PYTHON_SITEDIR)
    set(_sitedir ${PYTHON_SITEDIR})
else()
    set(_sitedir lib/python)
endif()

# install the hoomd_analyze python module
install(DIRECTORY freud
    DESTINATION ${_sitedir}
    PATTERN "*.pyc" EXCLUDE
    PATTERN "*.pyo" EXCLUDE
    )

# create a convenience target for editing .py files in IDEs
file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/freud/*.py)
add_custom_target(freud_py SOURCES ${SRCS})
