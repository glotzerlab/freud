# install the hoomd_analyze python module
install(DIRECTORY freud
    DESTINATION .
    PATTERN "*.pyc" EXCLUDE
    PATTERN "*.pyo" EXCLUDE
    PATTERN "*.pxd" EXCLUDE
    PATTERN "*.cpp" EXCLUDE
    PATTERN "*.pyx" EXCLUDE
    PATTERN "*.pxi" EXCLUDE
    )

# create a convenience target for editing .py files in IDEs
file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/freud/*.py)
add_custom_target(freud_py SOURCES ${SRCS})
