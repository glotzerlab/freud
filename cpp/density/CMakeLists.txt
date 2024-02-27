add_library(
  _density OBJECT
  CorrelationFunction.h
  CorrelationFunction.cc
  ContinuousCoordination.h
  ContinuousCoordination.cc
  GaussianDensity.h
  GaussianDensity.cc
  LocalDensity.h
  LocalDensity.cc
  RDF.h
  RDF.cc
  SphereVoxelization.h
  SphereVoxelization.cc)

target_link_libraries(_density PUBLIC TBB::tbb)

# We treat the extern folder as a SYSTEM library to avoid getting any diagnostic
# information from it. In particular, this avoids clang-tidy throwing errors due
# to any issues in external code.
target_include_directories(_density SYSTEM PUBLIC ${PROJECT_SOURCE_DIR}/extern/)
