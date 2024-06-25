# create the target and add the needed properties
nanobind_add_module(_box ${freud_nanobind_linking} module-box.cc export_Box.cc)
target_link_libraries(_box PUBLIC libfreud)

if(APPLE)
  set_target_properties(_box PROPERTIES INSTALL_RPATH "@loader_path")
else()
  set_target_properties(_box PROPERTIES INSTALL_RPATH "\$ORIGIN")
endif()

if(_using_conda)
  set_target_properties(_box PROPERTIES INSTALL_RPATH_USE_LINK_PATH True)
endif()

# install
install(TARGETS _box DESTINATION freud)
