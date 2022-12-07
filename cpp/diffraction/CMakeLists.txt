add_library(
  _diffraction OBJECT
  StructureFactor.h
  StaticStructureFactor.h
  StaticStructureFactor.cc
  StaticStructureFactorDebye.h
  StaticStructureFactorDebye.cc
  StructureFactorDirect.h
  StructureFactorDirect.cc
  StaticStructureFactorDirect.h
  StaticStructureFactorDirect.cc)
  #IntermediateScattering.h
  #IntermediateScattering.cc)

target_link_libraries(_diffraction PUBLIC TBB::tbb)

target_include_directories(_diffraction PUBLIC ${PROJECT_SOURCE_DIR}/extern/)
