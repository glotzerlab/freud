set(util_extension_headers export_ManagedArray.h export_Vector.h)

nanobind_add_module(_util module-util.cc
                    ${util_extension_headers})
target_link_libraries(_util PUBLIC freud)

target_set_install_rpath(_util)

# install
install(TARGETS _util DESTINATION freud)
