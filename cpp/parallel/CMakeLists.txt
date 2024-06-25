nanobind_add_module(_parallel NB_SHARED module-parallel.cc)
target_link_libraries(_parallel PUBLIC libfreud)

target_set_install_rpath(_parallel)

# install
install(TARGETS _parallel DESTINATION freud)
