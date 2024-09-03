// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "NeighborComputeFunctional.h"
#include <memory>
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "VectorMath.h"

/*! \file NeighborComputeFunctional.h
    \brief Implements logic for generic looping over neighbors and applying a compute function.
*/

namespace freud { namespace locality {

std::shared_ptr<NeighborList> makeDefaultNlist(const std::shared_ptr<NeighborQuery>& nq,
                                               const std::shared_ptr<NeighborList>& nlist,
                                               const vec3<float>* query_points, unsigned int num_query_points,
                                               QueryArgs qargs)
{
    std::shared_ptr<NeighborList> return_nlist;
    if (nlist == nullptr)
    {
        auto nqiter(nq->query(query_points, num_query_points, qargs));
        return_nlist = nqiter->toNeighborList();
    }
    else
    {
        return_nlist = nlist;
    }
    return_nlist->validate(num_query_points, nq->getNPoints());
    return return_nlist;
}

}; }; // end namespace freud::locality
