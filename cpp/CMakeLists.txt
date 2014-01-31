######################
# Perform setup to make a target into a functional python module
macro(setup_pymodule target)
set_target_properties(${target} PROPERTIES PREFIX "")
string(TOUPPER ${BOOST_PYTHON_COMPONENT} UPPER_BOOST_PYTHON_COMPONENT )
target_link_libraries(${target} ${Boost_${UPPER_BOOST_PYTHON_COMPONENT}_LIBRARY} ${PYTHON_LIBRARIES} ${TBB_LIBRARY})
endmacro(setup_pymodule)
#####################

######################
# Compile a list of ispc files
# this is loosly based on FindCUDA.cmake
macro(compile_ispc generated_files)
# reset the output variable
set(_compile_ispc_generated_files "")

# Iterate over the macro arguments and create custom
# commands for all the .ispc files.
foreach(file ${ARGN})
    get_filename_component( basename ${file} NAME )
    set(generated_file_path ${CMAKE_CURRENT_BINARY_DIR})
    set(generated_file_basename "${basename}")
    set(generated_obj_file "${generated_file_path}/${generated_file_basename}.o")
    set(generated_header_file "${generated_file_path}/${generated_file_basename}.h")

    # Don't add CMAKE_CURRENT_SOURCE_DIR if the path is already an absolute path.
    get_filename_component(file_path "${file}" PATH)
    if(IS_ABSOLUTE "${file_path}")
        set(source_file "${file}")
    else()
        set(source_file "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
    endif()
    
    add_custom_command(OUTPUT ${generated_obj_file} ${generated_header_file}
                       IMPLICIT_DEPENDS CXX
                       DEPENDS ${source_file}
                       COMMAND ${ISPC_EXECUTABLE} ${source_file} -o ${generated_obj_file} -h ${generated_header_file}
                      )
    
    set_source_files_properties(${generated_obj_file} PROPERTIES GENERATED TRUE)
    list(APPEND _compile_ispc_generated_files ${generated_obj_file})
endforeach()

# return the result
set(${generated_files} ${_compile_ispc_generated_files})
endmacro(compile_ispc)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/trajectory
                    ${CMAKE_CURRENT_SOURCE_DIR}/util
                    ${CMAKE_CURRENT_SOURCE_DIR}/locality
                    ${CMAKE_CURRENT_SOURCE_DIR}/cluster
                    ${CMAKE_CURRENT_SOURCE_DIR}/density
                    ${CMAKE_CURRENT_SOURCE_DIR}/voronoi
                    ${CMAKE_CURRENT_SOURCE_DIR}/kspace
                    ${CMAKE_CURRENT_SOURCE_DIR}/order
                    ${CMAKE_CURRENT_SOURCE_DIR}/interface
                    ${CMAKE_CURRENT_SOURCE_DIR}/sphericalharmonicorderparameters
                    ${CMAKE_CURRENT_SOURCE_DIR}/viz
                    ${CMAKE_CURRENT_SOURCE_DIR}/parallel
                    ${CMAKE_CURRENT_BINARY_DIR}
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
            density/LocalDensity.h
            density/LocalDensity.cc
            voronoi/VoronoiBuffer.h
            voronoi/VoronoiBuffer.cc
            kspace/kspace.h
            kspace/kspace.cc
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
            interface/InterfaceMeasure.h
            interface/InterfaceMeasure.h
            sphericalharmonicorderparameters/LocalQl.h
            sphericalharmonicorderparameters/LocalQl.cc
            sphericalharmonicorderparameters/LocalWl.h
            sphericalharmonicorderparameters/LocalWl.cc
            sphericalharmonicorderparameters/SolLiq.h
            sphericalharmonicorderparameters/SolLiq.cc
            viz/colormap.h
            viz/colormap.cc
            viz/colorutil.h
            viz/colorutil.cc
            viz/triangles.h
            viz/triangles.cc
            viz/split.h
            viz/split.cc
            parallel/tbb_config.h
            parallel/tbb_config.cc
            )
set_source_files_properties(${SOURCES} PROPERTIES COMPILE_DEFINITIONS NO_IMPORT_ARRAY)

compile_ispc(ispc_generated_files viz/colorutil.ispc)

add_library(_freud MODULE ${ispc_generated_files} ${SOURCES} util/num_util.cc util/num_util.h module.cc)
setup_pymodule(_freud)

if (PYTHON_SITEDIR)
    set(_sitedir ${PYTHON_SITEDIR})
else()
    set(_sitedir lib/python)
endif()

INSTALL(TARGETS _freud
  LIBRARY DESTINATION ${_sitedir}
  )
