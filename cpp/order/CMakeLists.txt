add_library(
    _order
    OBJECT
    Cubatic.cc
    Cubatic.h
    HexaticTranslational.cc
    HexaticTranslational.h
    Nematic.cc
    Nematic.h
    RotationalAutocorrelation.cc
    RotationalAutocorrelation.h
    SolidLiquid.cc
    SolidLiquid.h
    Steinhardt.cc
    Steinhardt.h
    Wigner3j.cc
    Wigner3j.h
    )

target_include_directories(
    _order
    PUBLIC ${CMAKE_SOURCE_DIR}/cpp/cluster
    )
target_link_libraries(_order TBB _diagonalize)
