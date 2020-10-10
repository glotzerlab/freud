add_library(
    _cluster
    OBJECT
    Cluster.h
    Cluster.cc
    ClusterProperties.h
    ClusterProperties.cc
    )

target_link_libraries(_cluster PUBLIC TBB)
target_include_directories(
    _cluster
    PUBLIC ${PROJECT_SOURCE_DIR}/extern/
    )
