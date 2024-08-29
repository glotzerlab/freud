nanobind_add_module(
  _pmft
  module-pmft.cc
  export-PMFTXY.cc
  export-PMFTXYZ.cc
  export-PMFTR12.cc
  export-PMFTXYT.cc)
target_link_libraries(_pmft PUBLIC freud)

target_set_install_rpath(_pmft)

install(TARGETS _pmft DESTINATION freud)
