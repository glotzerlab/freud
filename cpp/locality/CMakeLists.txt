set(locality_extension_headers export_PeriodicBuffer.h
                               export_NeighborQuery.h)

# create the target and add the needed properties
nanobind_add_module(_locality SHARED module-locality.cc ${locality_extension_headers})
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
