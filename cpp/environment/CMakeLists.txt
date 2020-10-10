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

# Note that at the moment the _util link doesn't appeear to propagate upward to
# things linking to _environment.
target_link_libraries(_environment PUBLIC TBB _util)
