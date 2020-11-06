add_library(_diffraction OBJECT StaticStructureFactor.h
                                StaticStructureFactor.cc)

target_include_directories(_diffraction
                           PUBLIC ${PROJECT_SOURCE_DIR}/cpp/density)
