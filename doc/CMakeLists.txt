find_package(Doxygen)

# disable DOT in doxygen if dot is not found
if(DOXYGEN_DOT_PATH)
    set(DOXYGEN_HAVE_DOT "YES")
else(DOXYGEN_DOT_PATH)
    set(DOXYGEN_HAVE_DOT "NO")
endif(DOXYGEN_DOT_PATH)

# generate a list of source files to depend on
file(GLOB_RECURSE _source_files
                  ${FREUD_SOURCE_DIR}/cpp/*.h
                  ${FREUD_SOURCE_DIR}/cpp/*.cc
                  ${FREUD_SOURCE_DIR}/cpp/*.cuh
                  ${FREUD_SOURCE_DIR}/cpp/*.cu
                  ${FREUD_SOURCE_DIR}/py/*.py
                  ${FREUD_SOURCE_DIR}/examples/*.cc
                  ${FREUD_SOURCE_DIR}/examples/*.py
                  ${CMAKE_CURRENT_SOURCE_DIR}/*.dox)

set (_dev_doc_sources  ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
                       )

# configure the doxygen file and target for the developer documentation
configure_file (${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/html/index.html
                    COMMAND ${DOXYGEN} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
                    DEPENDS ${_source_files}
                            ${_dev_doc_sources}
                            ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)

add_custom_target (doc ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/html/index.html
                           SOURCES ${_dev_doc_sources})

