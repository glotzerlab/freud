# create the target and add the needed properties
nanobind_add_module(_box NB_SHARED module-box.cc export_Box.cc)
target_link_libraries(_box PUBLIC libfreud)

target_set_install_rpath(_box)

# install
install(TARGETS _box DESTINATION freud)
