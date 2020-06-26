#include "NeighborComputeFunctional.h"

/*! \file NeighborComputeFunctional.h
    \brief Implements logic for generic looping over neighbors and applying a compute function.
*/

namespace freud { namespace locality {

NeighborList makeDefaultNlist(const NeighborQuery* nq, const NeighborList* nlist,
                              const vec3<float>* query_points, unsigned int num_query_points,
                              locality::QueryArgs qargs)
{
    if (nlist == NULL)
    {
        auto nqiter(nq->query(query_points, num_query_points, qargs));
        nlist = nqiter->toNeighborList();
    }
    // Ideally we wouldn't allocate a new NeighborList at all, but we don't want to force
    // calling code to manage the memory of a raw pointer, so we prefer to return by value
    // here, and the toNeighborList function has to return a raw pointer to work with the
    // Cython interface of freud. Therefore, for now we're sticking with this suboptimal
    // solution until we can refactor these internalsa little more cleanly.
    locality::NeighborList new_nlist = NeighborList(*nlist);
    new_nlist.validate(num_query_points, nq->getNPoints());
    delete nlist;
    return new_nlist;
}

}; }; // end namespace freud::locality
