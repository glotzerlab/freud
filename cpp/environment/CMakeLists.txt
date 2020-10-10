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

target_link_libraries(
    _environment
    PUBLIC TBB
    PRIVATE _util
    )
target_include_directories(
    _environment
    PUBLIC ${PROJECT_SOURCE_DIR}/extern/
    )
