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
    )

target_link_libraries(_locality tbb)
