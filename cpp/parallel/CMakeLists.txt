nanobind_add_module(_parallel NB_SHARED module-parallel.cc)
target_link_libraries(_parallel PUBLIC libfreud)

if(APPLE)
  set_target_properties(_parallel PROPERTIES INSTALL_RPATH "@loader_path")
else()
  set_target_properties(_parallel PROPERTIES INSTALL_RPATH "\$ORIGIN")
endif()

if(_using_conda)
  set_target_properties(_parallel PROPERTIES INSTALL_RPATH_USE_LINK_PATH True)
endif()

# install
install(TARGETS _parallel DESTINATION freud)
