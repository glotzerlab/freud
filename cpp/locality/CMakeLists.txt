set(locality_extension_sources
    export_NeighborQuery.cc
    export_NeighborList.cc
    export_Filter.cc
    export_PeriodicBuffer.cc
    export_Voronoi.cc)

# create the target and add the needed properties
nanobind_add_module(_locality SHARED module-locality.cc
                    ${locality_extension_sources})
target_link_libraries(_locality PUBLIC libfreud)

if(APPLE)
  set_target_properties(_locality PROPERTIES INSTALL_RPATH "@loader_path")
else()
  set_target_properties(_locality PROPERTIES INSTALL_RPATH "\$ORIGIN")
endif()

if(_using_conda OR DEFINED ENV{CIBUILDWHEEL})
  set_target_properties(_locality PROPERTIES INSTALL_RPATH_USE_LINK_PATH True)
endif()

# install
install(TARGETS _locality DESTINATION freud)
