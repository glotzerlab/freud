add_library(
  _diffraction OBJECT
  StaticStructureFactorDebye.h StaticStructureFactorDebye.cc
  StaticStructureFactorDirect.h StaticStructureFactorDirect.cc)

target_include_directories(
  _diffraction
  PUBLIC ${PROJECT_SOURCE_DIR}/cpp/density
  PUBLIC ${PROJECT_SOURCE_DIR}/extern/)
