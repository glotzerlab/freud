add_library(
    _cluster
    OBJECT
    Cluster.h
    Cluster.cc
    ClusterProperties.h
    ClusterProperties.cc
    )

target_link_libraries(_cluster TBB)
