// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "NeighborQuery.h"

namespace freud { namespace locality {

const NeighborBond NeighborQueryIterator::ITERATOR_TERMINATOR(-1, -1, 0);

const QueryArgs::QueryType QueryArgs::DEFAULT_MODE(QueryArgs::QueryType::none);
const int QueryArgs::DEFAULT_NUM_NEIGH(-1);
const float QueryArgs::DEFAULT_R_MAX(-1);
const float QueryArgs::DEFAULT_SCALE(-1);
const bool QueryArgs::DEFAULT_EXCLUDE_II(false);
}; }; // end namespace freud::locality
