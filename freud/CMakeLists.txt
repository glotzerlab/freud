
# make CYTHON_SOURCES available for all following CMakeLists.txt files
# set(CYTHON_SOURCES "${CYTHON_SOURCES}" PARENT_SCOPE)

# # install the hoomd_analyze python module
# install(DIRECTORY .
#     DESTINATION .
#     PATTERN "*.pyc" EXCLUDE
#     PATTERN "*.pyo" EXCLUDE
#     )

# # create a convenience target for editing .py files in IDEs
# file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
# add_custom_target(freud_py SOURCES ${SRCS})
