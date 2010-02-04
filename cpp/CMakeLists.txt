######################
# Perform setup to make a target into a functional python module
macro(setup_pymodule target)
set_target_properties(${target} PROPERTIES PREFIX "")
target_link_libraries(${target} ${Boost_PYTHON_LIBRARY} ${PYTHON_LIBRARIES})
endmacro(setup_pymodule)
#####################

set(SOURCES trajectory.cc)

set_source_files_properties(${SOURCES} PROPERTIES COMPILE_DEFINITIONS NO_IMPORT_ARRAY)
add_library(_freud MODULE ${SOURCES} num_util.cc module.cc)
setup_pymodule(_freud)
