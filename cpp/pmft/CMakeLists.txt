nanobind_add_module(_pmft ${freud_nanobind_linking} module-pmft.cc export-PMFTXY.cc)
target_link_libraries(_pmft PUBLIC libfreud)

target_set_install_rpath(_pmft)

install(TARGETS _pmft DESTINATION freud)
