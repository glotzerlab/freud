######################
# Perform setup to make a target into a functional python module
macro(setup_pymodule target)
set_target_properties(${target} PROPERTIES PREFIX "")
string(TOUPPER ${BOOST_PYTHON_COMPONENT} UPPER_BOOST_PYTHON_COMPONENT )
target_link_libraries(${target} ${Boost_${UPPER_BOOST_PYTHON_COMPONENT}_LIBRARY} ${PYTHON_LIBRARIES})
endmacro(setup_pymodule)
#####################

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/trajectory
                    ${CMAKE_CURRENT_SOURCE_DIR}/util
                    ${CMAKE_CURRENT_SOURCE_DIR}/locality
                    ${CMAKE_CURRENT_SOURCE_DIR}/cluster
                    ${CMAKE_CURRENT_SOURCE_DIR}/density
                    ${CMAKE_CURRENT_SOURCE_DIR}/order
                    ${CMAKE_CURRENT_SOURCE_DIR}/interface
                    )

set(SOURCES trajectory/trajectory.cc
            trajectory/trajectory.h
            trajectory/DCDLoader.cc
            trajectory/DCDLoader.h
            locality/LinkCell.cc
            locality/LinkCell.h
            density/RDF.cc
            density/RDF.h
            density/GaussianDensity.cc
            density/GaussianDensity.h
            cluster/Cluster.h
            cluster/Cluster.cc
            cluster/ClusterProperties.h
            cluster/ClusterProperties.cc
            order/HexOrderParameter.h
            order/HexOrderParameter.cc
            util/Index1D.h
            util/HOOMDMath.h
            molfile/dcdplugin.cc
            molfile/vmdplugin.h
            molfile/endianswap.h
            molfile/fastio.h
            molfile/largefiles.h
            molfile/molfile_plugins.h
            molfile/molfile_plugin.h
            interface/InterfaceMeasure.cc
            interface/InterfaceMeasure.h)

set_source_files_properties(${SOURCES} PROPERTIES COMPILE_DEFINITIONS NO_IMPORT_ARRAY)
add_library(_freud MODULE ${SOURCES} util/num_util.cc util/num_util.h module.cc)
setup_pymodule(_freud)

if (PYTHON_SITEDIR)
    set(_sitedir ${PYTHON_SITEDIR})
else()
    set(_sitedir lib/python)
endif()

INSTALL(TARGETS _freud
  LIBRARY DESTINATION ${_sitedir}
  )
