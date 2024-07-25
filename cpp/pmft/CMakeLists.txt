set(pmft_extension_sources export_PMFT.h)

nanobind_add_module(_pmft ${freud_nanobind_linking} module-pmft.cc
                    ${pmft_extension_sources})
target_link_libraries(_pmft PUBLIC libfreud)

target_set_install_rpath(_pmft)

install(TARGETS _pmft DESTINATION freud)
