######################
# Perform setup to make a target into a functional python module
macro(setup_pymodule target)
set_target_properties(${target} PROPERTIES PREFIX "")

if (APPLE)
    set_target_properties(${target} PROPERTIES SUFFIX ".so")
endif(APPLE)

target_link_libraries(${target} ${TBB_LIBRARY})

if (APPLE)
    target_link_libraries(${target} "-undefined dynamic_lookup")
endif()

fix_tbb_rpath(${target})
fix_conda_python(${target})
endmacro(setup_pymodule)
#####################

file(GLOB_RECURSE PXIS ${CMAKE_CURRENT_SOURCE_DIR}/../freud/*.pxi)
file(GLOB_RECURSE PYXS ${CMAKE_CURRENT_SOURCE_DIR}/../freud/*.pyx)

set(PYX_CPPS ${CMAKE_CURRENT_SOURCE_DIR}/../freud/_freud.cpp)

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
                    ${CMAKE_CURRENT_SOURCE_DIR}/box
                    ${CMAKE_CURRENT_SOURCE_DIR}/bond
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
                    ${CMAKE_CURRENT_SOURCE_DIR}/parallel
                    ${CMAKE_CURRENT_SOURCE_DIR}/registration
                    ${CMAKE_CURRENT_BINARY_DIR}
                    )

set(FREUD_SOURCES
            box/box.h
            box/box.cc
            bond/BondingAnalysis.h
            bond/BondingAnalysis.cc
            bond/BondingR12.h
            bond/BondingR12.cc
            bond/BondingXY2D.h
            bond/BondingXY2D.cc
            bond/BondingXYT.h
            bond/BondingXYT.cc
            bond/BondingXYZ.h
            bond/BondingXYZ.cc
            locality/LinkCell.cc
            locality/LinkCell.h
            locality/NearestNeighbors.h
            locality/NearestNeighbors.cc
            locality/NeighborList.cc
            density/CorrelationFunction.h
            density/CorrelationFunction.cc
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
            order/CubaticOrderParameter.h
            order/CubaticOrderParameter.cc
            order/NematicOrderParameter.h
            order/NematicOrderParameter.cc
            order/AngularSeparation.h
            order/AngularSeparation.cc
            order/BondOrder.h
            order/BondOrder.cc
            order/LocalDescriptors.h
            order/LocalDescriptors.cc
            util/Index1D.h
            util/HOOMDMath.h
            util/HOOMDMatrix.cc
            interface/InterfaceMeasure.cc
            interface/InterfaceMeasure.h
            order/Pairing2D.cc
            order/Pairing2D.h
            pmft/PMFT.cc
            pmft/PMFT.h
            pmft/PMFTXYZ.cc
            pmft/PMFTXYZ.h
            pmft/PMFTXY2D.cc
            pmft/PMFTXY2D.h
            pmft/PMFTR12.cc
            pmft/PMFTR12.h
            pmft/PMFTXYT.cc
            pmft/PMFTXYT.h
            shapesplit/shapesplit.cc
            shapesplit/shapesplit.h
            interface/InterfaceMeasure.h
            order/Steinhardt.h
            order/Steinhardt.cc
            order/LocalWl.h
            order/LocalWl.cc
            order/SolLiq.h
            order/SolLiq.cc
            order/MatchEnv.h
            order/MatchEnv.cc
            order/wigner3j.cc
            order/wigner3j.h
            parallel/tbb_config.h
            parallel/tbb_config.cc
            registration/brute_force.h
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
