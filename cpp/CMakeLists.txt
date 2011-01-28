######################
# Perform setup to make a target into a functional python module
macro(setup_pymodule target)
set_target_properties(${target} PROPERTIES PREFIX "")
target_link_libraries(${target} ${Boost_PYTHON_LIBRARY} ${PYTHON_LIBRARIES})
endmacro(setup_pymodule)
#####################

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/trajectory
                    ${CMAKE_CURRENT_SOURCE_DIR}/util
                    ${CMAKE_CURRENT_SOURCE_DIR}/locality)

set(SOURCES trajectory/trajectory.cc
            trajectory/trajectory.h
            trajectory/DCDLoader.cc
            trajectory/DCDLoader.h
            locality/LinkCell.cc
            locality/LinkCell.h
            locality/RDF.cc
            locality/RDF.h
            util/Index1D.h
            util/HOOMDMath.h
            molfile/dcdplugin.cc
            molfile/vmdplugin.h
            molfile/endianswap.h
            molfile/fastio.h
            molfile/largefiles.h
            molfile/molfile_plugins.h
            molfile/molfile_plugin.h)

set_source_files_properties(${SOURCES} PROPERTIES COMPILE_DEFINITIONS NO_IMPORT_ARRAY)
add_library(_freud MODULE ${SOURCES} util/num_util.cc util/num_util.h module.cc)
setup_pymodule(_freud)
