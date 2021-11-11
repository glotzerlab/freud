add_library(
  _diffraction OBJECT
  StaticStructureFactor.h StaticStructureFactor.cc StaticStructureFactorDebye.h
  StaticStructureFactorDebye.cc StaticStructureFactorDirect.h
  StaticStructureFactorDirect.cc)

target_link_libraries(_diffraction PUBLIC TBB::tbb)

target_include_directories(_diffraction PUBLIC ${PROJECT_SOURCE_DIR}/extern/)
