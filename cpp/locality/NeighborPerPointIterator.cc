// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "NeighborPerPointIterator.h"

namespace freud { namespace locality {

NeighborPerPointIterator::NeighborPerPointIterator() {}

NeighborPerPointIterator::NeighborPerPointIterator(unsigned int query_point_idx) : m_query_point_idx(query_point_idx)
{}

NeighborPerPointIterator::~NeighborPerPointIterator() {}


const NeighborBond NeighborPerPointIterator::ITERATOR_TERMINATOR(-1, -1, 0);

}; }; // end namespace freud::locality
