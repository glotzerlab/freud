nanobind_add_module(_parallel module-parallel.cc)
target_link_libraries(_parallel PUBLIC freud TBB::tbb)

target_set_install_rpath(_parallel)

# install
install(TARGETS _parallel DESTINATION freud)
