set(VOROPP_SOURCE_DIR
    "${PROJECT_SOURCE_DIR}/extern/voro++/src"
    )

add_library(
    _locality
    OBJECT
    AABB.h
    AABBQuery.cc
    AABBQuery.h
    AABBTree.h
    BondHistogramCompute.h
    CMakeLists.txt
    LinkCell.cc
    LinkCell.h
    NeighborBond.h
    NeighborComputeFunctional.cc
    NeighborComputeFunctional.h
    NeighborList.cc
    NeighborList.h
    NeighborPerPointIterator.cc
    NeighborPerPointIterator.h
    NeighborQuery.cc
    NeighborQuery.h
    PeriodicBuffer.cc
    PeriodicBuffer.h
    RawPoints.h
    Voronoi.cc
    Voronoi.h

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
    ${VOROPP_SOURCE_DIR}/container_prd.cc
    )

target_link_libraries(_locality PUBLIC TBB)
target_include_directories(
    _locality
    PUBLIC ${PROJECT_SOURCE_DIR}/extern/
    )
