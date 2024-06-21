set(VOROPP_SOURCE_DIR "${PROJECT_SOURCE_DIR}/extern/voro++/src")

set(locality_sources
    locality/AABBQuery.cc
    locality/FilterRAD.cc
    locality/FilterSANN.cc
    locality/LinkCell.cc
    locality/NeighborComputeFunctional.cc
    locality/NeighborList.cc
    locality/PeriodicBuffer.cc
    locality/Voronoi.cc
    # For now, compile voro++ object in directly.
    ${VOROPP_SOURCE_DIR}/cell.cc
    ${VOROPP_SOURCE_DIR}/common.cc
    ${VOROPP_SOURCE_DIR}/container.cc
    ${VOROPP_SOURCE_DIR}/unitcell.cc
    ${VOROPP_SOURCE_DIR}/v_compute.cc
    ${VOROPP_SOURCE_DIR}/c_loops.cc
    ${VOROPP_SOURCE_DIR}/v_base.cc
    ${VOROPP_SOURCE_DIR}/wall.cc
    ${VOROPP_SOURCE_DIR}/pre_container.cc
    ${VOROPP_SOURCE_DIR}/container_prd.cc)

set(locality_headers
    locality/AABB.h
    locality/AABBQuery.h
    locality/AABBTree.h
    locality/BondHistogramCompute.h
    locality/Filter.h
    locality/FilterRAD.h
    locality/FilterSANN.h
    locality/LinkCell.h
    locality/NeighborBond.h
    locality/NeighborList.h
    localty/NeighborPerPointIterator.h
    locality/NeighborQuery.h
    locality/PeriodicBuffer.h
    locality/RawPoints.h
    locality/Voronoi.h)
