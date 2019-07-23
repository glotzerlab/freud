#include "NeighborComputeFunctional.h"

namespace freud { namespace locality {

std::shared_ptr<NeighborIterator> getNeighborIterator(
    const NeighborQuery* neighbor_query, const vec3<float>* query_points, unsigned int n_query_points,
    QueryArgs qargs, const NeighborList* nlist)
{
    if (nlist != NULL)
    {
        return std::make_shared<NeighborListNeighborIterator>(nlist);
    }
    else
    {
        return std::make_shared<NeighborQueryNeighborIterator>(neighbor_query, query_points, n_query_points, qargs);
    }
}

}; };
