// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "NeighborComputeFunctional.h"

/*! \file NeighborComputeFunctional.h
    \brief Implements logic for generic looping over neighbors and applying a compute function.
*/

namespace freud { namespace locality {

void makeDefaultNlist(std::shared_ptr<NeighborQuery> nq, std::shared_ptr<NeighborList> nlist,
        const vec3<float>* query_points, unsigned int num_query_points, QueryArgs qargs)
{
    if (nlist == nullptr)
    {
        auto nqiter(nq->query(query_points, num_query_points, qargs));
        nlist = nqiter->toNeighborList();
    }
    nlist->validate(num_query_points, nq->getNPoints());
}

}; }; // end namespace freud::locality
