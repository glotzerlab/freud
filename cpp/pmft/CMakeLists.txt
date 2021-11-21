add_library(
  _pmft OBJECT
  PMFTR12.h
  PMFTR12.cc
  PMFTXY.h
  PMFTXY.cc
  PMFTXYT.h
  PMFTXYT.cc
  PMFTXYZ.h
  PMFTXYZ.cc)

target_link_libraries(_pmft PUBLIC TBB::tbb)
