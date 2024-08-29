# create the target and add the needed properties
nanobind_add_module(_box module-box.cc export_Box.cc)
target_link_libraries(_box PUBLIC TBB::tbb)

target_set_install_rpath(_box)

# install
install(TARGETS _box DESTINATION freud)
