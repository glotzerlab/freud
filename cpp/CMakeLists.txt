######################
# Perform setup to make a target into a functional python module
macro(setup_pymodule target)
set_target_properties(${target} PROPERTIES PREFIX "")
target_link_libraries(${target} ${PYTHON_LIBRARIES} ${TBB_LIBRARY})
endmacro(setup_pymodule)
#####################

file(GLOB_RECURSE PXIS ${CMAKE_CURRENT_SOURCE_DIR}/../freud/*.pxi)
file(GLOB_RECURSE PYXS ${CMAKE_CURRENT_SOURCE_DIR}/../freud/*.pyx)

file(GLOB_RECURSE PYX_CPPS ${CMAKE_CURRENT_SOURCE_DIR}/../freud/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../freud/*.c)

set(ENABLE_CYTHON OFF CACHE BOOL "Regenerate cython sources")
if(ENABLE_CYTHON)
  foreach(pyx IN LISTS PYXS)
    file(RELATIVE_PATH pyx_rel ${CMAKE_CURRENT_SOURCE_DIR} ${pyx})
    string(REPLACE ".pyx" ".cpp" cpp_rel "${pyx_rel}")
    set(cpp_out "${CMAKE_CURRENT_SOURCE_DIR}/${cpp_rel}")
    add_custom_command(OUTPUT ${cpp_out}
                       COMMAND ${CYTHON_EXECUTABLE} -X boundscheck=False -o ${cpp_out} -w ${CMAKE_CURRENT_SOURCE_DIR}/../freud --cplus ${pyx}
                       DEPENDS ${pyx} ${PXIS})
    set_source_files_properties(${cpp_out} PROPERTIES GENERATED TRUE)
    list(APPEND CYTHON_SOURCES ${cpp_out})
  endforeach(pyx in LISTS PYXS)
else(ENABLE_CYTHON)
  set(CYTHON_SOURCES ${PYX_CPPS})
endif(ENABLE_CYTHON)

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
                    ${CMAKE_CURRENT_SOURCE_DIR}/pmft
                    ${CMAKE_CURRENT_SOURCE_DIR}/shapesplit
                    ${CMAKE_CURRENT_SOURCE_DIR}/viz
                    ${CMAKE_CURRENT_SOURCE_DIR}/parallel
                    ${CMAKE_CURRENT_BINARY_DIR}
                    )

set(FREUD_SOURCES
            trajectory/trajectory.h
            trajectory/DCDLoader.cc
            trajectory/DCDLoader.h
            locality/LinkCell.cc
            locality/LinkCell.h
            locality/NearestNeighbors.h
            locality/NearestNeighbors.cc
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
            order/TransOrderParameter.h
            order/TransOrderParameter.cc
            order/BondOrder.h
            order/BondOrder.cc
            order/CubaticOrderParameter.h
            order/CubaticOrderParameter.cc
            order/EntropicBonding.h
            order/EntropicBonding.cc
            order/LocalDescriptors.h
            order/LocalDescriptors.cc
            util/Index1D.h
            util/HOOMDMath.h
            util/HOOMDMatrix.cc
            molfile/dcdplugin.cc
            molfile/vmdplugin.h
            molfile/endianswap.h
            molfile/fastio.h
            molfile/largefiles.h
            molfile/molfile_plugins.h
            molfile/molfile_plugin.h
            interface/InterfaceMeasure.cc
            interface/InterfaceMeasure.h
            order/Pairing2D.cc
            order/Pairing2D.h
            pmft/PMFXYZ.cc
            pmft/PMFXYZ.h
            pmft/PMFXY2D.cc
            pmft/PMFXY2D.h
            pmft/PMFTXYT2D.cc
            pmft/PMFTXYT2D.h
            pmft/PMFTXYTP2D.cc
            pmft/PMFTXYTP2D.h
            pmft/PMFTXYTM2D.cc
            pmft/PMFTXYTM2D.h
            pmft/PMFTRPM.cc
            pmft/PMFTRPM.h
            pmft/PMFTR12.cc
            pmft/PMFTR12.h
            shapesplit/shapesplit.cc
            shapesplit/shapesplit.h
            interface/InterfaceMeasure.h
            order/LocalQl.h
            order/LocalQl.cc
            order/LocalQlNear.h
            order/LocalQlNear.cc
            order/LocalWl.h
            order/LocalWl.cc
            order/LocalWlNear.h
            order/LocalWlNear.cc
            order/SolLiq.h
            order/SolLiq.cc
            order/SolLiqNear.h
            order/SolLiqNear.cc
            order/wigner3j.cc
            order/wigner3j.h
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

foreach(src IN LISTS FREUD_SOURCES)
  list(APPEND SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${src})
endforeach(src IN LISTS FREUD_SOURCES)

set_source_files_properties(${SOURCES} PROPERTIES COMPILE_DEFINITIONS NO_IMPORT_ARRAY)
add_library(_freud MODULE ${SOURCES} ${CYTHON_SOURCES})
setup_pymodule(_freud)

INSTALL(TARGETS _freud
  LIBRARY DESTINATION freud
  )
