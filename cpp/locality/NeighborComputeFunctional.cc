#include "NeighborComputeFunctional.h"

/*! \file NeighborComputeFunctional.h
    \brief Implements logic for generic looping over neighbors and applying a compute function.
*/

namespace freud { namespace locality {


//! Make a default NeighborList object to use.
/*! This function makes a NeighborList from the provided NeighborQuery object
 * if the provided NeighborList is NULL. Otherwise, it simply returns a copy of
 * the provided NeighborList.
 */
NeighborList makeDefaultNlist(const NeighborQuery *nq, const NeighborList
        *nlist, const vec3<float>* query_points, unsigned int num_query_points,
        locality::QueryArgs qargs)
{
    if (nlist == NULL)
    {
        auto nqiter(nq->query(query_points, num_query_points, qargs));
        nlist = nqiter->toNeighborList();
    }
    locality::NeighborList new_nlist = NeighborList(*nlist);
    new_nlist.validate(num_query_points, nq->getNPoints());
    return new_nlist;
}

}; }; // end namespace freud::locality
