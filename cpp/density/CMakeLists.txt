add_library(
  _density OBJECT
  CorrelationFunction.h
  CorrelationFunction.cc
  GaussianDensity.h
  GaussianDensity.cc
  LocalDensity.h
  LocalDensity.cc
  RDF.h
  RDF.cc
  SphereVoxelization.h
  SphereVoxelization.cc)

target_link_libraries(_density PUBLIC TBB::tbb)
