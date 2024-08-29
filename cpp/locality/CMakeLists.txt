set(locality_extension_sources
    export_BondHistogramCompute.cc export_NeighborQuery.cc
    export_NeighborList.cc export_Filter.cc export_PeriodicBuffer.cc
    export_Voronoi.cc)

# create the target and add the needed properties
nanobind_add_module(_locality module-locality.cc
                    ${locality_extension_sources})
target_link_libraries(_locality PUBLIC freud TBB::tbb)

target_set_install_rpath(_locality)

# install
install(TARGETS _locality DESTINATION freud)
