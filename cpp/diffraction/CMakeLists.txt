add_library(
  _diffraction OBJECT
  StaticStructureFactorDebye.h StaticStructureFactorDebye.cc
  StaticStructureFactorUtil.h StaticStructureFactorUtil.cc)

target_include_directories(_diffraction
                           PUBLIC ${PROJECT_SOURCE_DIR}/cpp/density)
