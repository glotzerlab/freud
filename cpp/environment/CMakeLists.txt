add_library(
    _environment
    OBJECT
    AngularSeparation.h
    AngularSeparation.cc
    BondOrder.h
    BondOrder.cc
    LocalBondProjection.h
    LocalBondProjection.cc
    LocalDescriptors.h
    LocalDescriptors.cc
    MatchEnv.h
    MatchEnv.cc
    )

target_link_libraries(_environment PUBLIC tbb _diagonalize)
