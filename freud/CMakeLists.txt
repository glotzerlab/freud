# ##############################################################################
# C++ component

# set the rpath for installing python extension modules
function(target_set_install_rpath _target)
  if(_using_conda)
    set_target_properties(${_target} PROPERTIES INSTALL_RPATH_USE_LINK_PATH
                                                True)
  endif()
endfunction()

set(VOROPP_SOURCE_DIR "${PROJECT_SOURCE_DIR}/extern/voro++/src")

# Use a static library for the main code. A Dynamic library would be more disk
# space efficient, but I was unable to resolve a RPATH issue where any use of
# `INSTALL_RPATH` would result in "terminated by signal SIGKILL" the moment
# Python attempts to load the library.
add_library(
  freud STATIC
  # box
  box/Box.h
  # cluster
  # cluster/Cluster.h
  # cluster/Cluster.cc
  # cluster/ClusterProperties.h
  # cluster/ClusterProperties.cc
  # density
  # density/CorrelationFunction.h
  # density/CorrelationFunction.cc
  # density/GaussianDensity.h
  # density/GaussianDensity.cc
  density/LocalDensity.h
  density/LocalDensity.cc
  density/RDF.h
  density/RDF.cc
  # density/SphereVoxelization.h
  # density/SphereVoxelization.cc
  # diffraction
  # diffraction/StaticStructureFactor.h
  # diffraction/StaticStructureFactor.cc
  # diffraction/StaticStructureFactorDebye.h
  # diffraction/StaticStructureFactorDebye.cc
  # diffraction/StaticStructureFactorDirect.h
  # diffraction/StaticStructureFactorDirect.cc
  # environment
  # environment/AngularSeparation.h
  # environment/AngularSeparation.cc
  # environment/BondOrder.h
  # environment/BondOrder.cc
  # environment/LocalBondProjection.h
  # environment/LocalBondProjection.cc
  # environment/LocalDescriptors.h
  # environment/LocalDescriptors.cc
  # environment/MatchEnv.h
  # environment/MatchEnv.cc
  # environment/Registration.h
  # locality
  locality/AABB.h
  locality/AABBQuery.cc
  locality/AABBQuery.h
  locality/AABBTree.h
  locality/BondHistogramCompute.h
  locality/Filter.h
  locality/FilterRAD.h
  locality/FilterRAD.cc
  locality/FilterSANN.h
  locality/FilterSANN.cc
  locality/LinkCell.h
  locality/LinkCell.cc
  locality/NeighborBond.h
  locality/NeighborList.h
  locality/NeighborPerPointIterator.h
  locality/NeighborComputeFunctional.cc
  locality/NeighborList.cc
  locality/NeighborQuery.h
  locality/PeriodicBuffer.h
  locality/PeriodicBuffer.cc
  locality/RawPoints.h
  locality/Voronoi.h
  locality/Voronoi.cc
  # Compile voro++ object in directly.
  ${VOROPP_SOURCE_DIR}/cell.cc
  ${VOROPP_SOURCE_DIR}/common.cc
  ${VOROPP_SOURCE_DIR}/container.cc
  ${VOROPP_SOURCE_DIR}/unitcell.cc
  ${VOROPP_SOURCE_DIR}/v_compute.cc
  ${VOROPP_SOURCE_DIR}/c_loops.cc
  ${VOROPP_SOURCE_DIR}/v_base.cc
  ${VOROPP_SOURCE_DIR}/wall.cc
  ${VOROPP_SOURCE_DIR}/pre_container.cc
  ${VOROPP_SOURCE_DIR}/container_prd.cc
  # order
  # order/ContinuousCoordination.h
  # order/ContinuousCoordination.cc
  # order/Cubatic.cc
  # order/Cubatic.h
  # order/HexaticTranslational.cc
  # order/HexaticTranslational.h
  # order/Nematic.cc
  # order/Nematic.h
  # order/RotationalAutocorrelation.cc
  # order/RotationalAutocorrelation.h
  # order/SolidLiquid.cc
  # order/SolidLiquid.h
  # order/Steinhardt.cc
  # order/Steinhardt.h
  # order/Wigner3j.cc
  # order/Wigner3j.h
  # parallel
  parallel/tbb_config.h
  parallel/tbb_config.cc
  # pmft
  pmft/PMFT.h
  pmft/PMFTXY.h
  pmft/PMFTXYZ.h
  pmft/PMFTR12.h
  pmft/PMFTXYT.h
  pmft/PMFTXY.cc
  pmft/PMFTXYZ.cc
  pmft/PMFTR12.cc
  pmft/PMFTXYT.cc
  # util
  util/BiMap.h
  util/Histogram.h
  util/ManagedArray.h
  util/ThreadStorage.h
  util/VectorMath.h
  util/utils.h
  util/diagonalize.h
  util/diagonalize.cc)

set_target_properties(freud PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(freud PUBLIC TBB::tbb)
target_include_directories(freud SYSTEM PUBLIC ${PROJECT_SOURCE_DIR}/extern/)

# cluster nanobind_add_module(_cluster cluster/...)
# target_link_libraries(_cluster PUBLIC freud TBB::tbb)
# target_set_install_rpath(_cluster)

# density nanobind_add_module(_density density/...)
# target_link_libraries(_density PUBLIC freud TBB::tbb)
# target_set_install_rpath(_density)

# diffraction nanobind_add_module(_diffraction diffraction/...)
# target_link_libraries(_diffraction PUBLIC freud TBB::tbb)
# target_set_install_rpath(_diffraction)

# environment nanobind_add_module(_environment environment/...)
# target_link_libraries(_environment PUBLIC freud TBB::tbb)
# target_set_install_rpath(_environment)

# box
nanobind_add_module(_box box/module-box.cc box/export-Box.cc box/export-Box.h)
target_link_libraries(_box PUBLIC TBB::tbb)
target_set_install_rpath(_box)

# locality
nanobind_add_module(
  _locality
  locality/module-locality.cc
  locality/export-BondHistogramCompute.cc
  locality/export-NeighborQuery.cc
  locality/export-NeighborList.cc
  locality/export-Filter.cc
  locality/export-PeriodicBuffer.cc
  locality/export-Voronoi.cc)
target_link_libraries(_locality PUBLIC freud TBB::tbb)
target_set_install_rpath(_locality)

# order nanobind_add_module(_order order/...) target_link_libraries(_order
# PUBLIC freud TBB::tbb) target_set_install_rpath(_order)

# parallel
nanobind_add_module(_parallel parallel/module-parallel.cc)
target_link_libraries(_parallel PUBLIC freud TBB::tbb)
target_set_install_rpath(_parallel)

# pmft
nanobind_add_module(
  _pmft pmft/module-pmft.cc pmft/export-PMFTXY.cc pmft/export-PMFTXYZ.cc
  pmft/export-PMFTR12.cc pmft/export-PMFTXYT.cc)
target_link_libraries(_pmft PUBLIC freud)
target_set_install_rpath(_pmft)

# density
nanobind_add_module(
  _density density/module-density.cc density/export-RDF.cc density/export-LocalDensity.cc)
target_link_libraries(_density PUBLIC freud)
target_set_install_rpath(_density)

# util
nanobind_add_module(_util util/module-util.cc util/export-ManagedArray.cc
                    util/export-Vector.cc util/export-Vector.h)
target_link_libraries(_util PUBLIC freud)
target_set_install_rpath(_util)

# ##############################################################################
# Python module
set(python_files
    __init__.py
    box.py
    data.py
    density.py
    errors.py
    locality.py
    parallel.py
    pmft.py
    plot.py
    util.py)
# cluster.py density.py diffraction.py environment.py order.py interface.py
# msd.py)

copy_files_to_build("${python_files}" "freud" "*.py")

# install
if(SKBUILD)
  install(FILES ${python_files} DESTINATION freud)
  install(TARGETS _box DESTINATION freud)
  # install(TARGETS _cluster DESTINATION freud) install(TARGETS _density
  # DESTINATION freud) install(TARGETS _diffraction DESTINATION freud)
  # install(TARGETS _environment DESTINATION freud)
  install(TARGETS _locality DESTINATION freud)
  # install(TARGETS _order DESTINATION freud)
  install(TARGETS _parallel DESTINATION freud)
  install(TARGETS _pmft DESTINATION freud)
  install(TARGETS _util DESTINATION freud)
endif()
