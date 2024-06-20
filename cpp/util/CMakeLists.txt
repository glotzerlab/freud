nanobind_add_module(_util NB_SHARED module-util.cc)
target_link_libraries(_util PUBLIC libfreud)

if(APPLE)
  set_target_properties(_util PROPERTIES INSTALL_RPATH "@loader_path")
else()
  set_target_properties(_util PROPERTIES INSTALL_RPATH "\$ORIGIN")
endif()

if(_using_conda OR DEFINED ENV{CIBUILDWHEEL})
  set_target_properties(_util PROPERTIES INSTALL_RPATH_USE_LINK_PATH True)
endif()

# install
install(TARGETS _util DESTINATION freud)
