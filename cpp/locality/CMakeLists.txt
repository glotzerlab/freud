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

target_link_libraries(_locality PUBLIC TBB voro++)
# Unclear if this is the best way to set the rpath (for voro++). Moreover, I'm
# suspecting it will be easier to just directly build all the voro++ files into
# it. Currently this solution is working, but rebuilding (running setup.py multiple
# times) prints some install_name_tool errors due to trying to delete rpaths. Note
# that these have no practical effect right now.
target_link_options(_locality INTERFACE -Wl,-rpath,${CMAKE_INSTALL_PREFIX}/lib)
