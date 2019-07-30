#include "NeighborComputeFunctional.h"

namespace freud { namespace locality {

std::shared_ptr<NeighborIterator> getNeighborIterator(const NeighborQuery* ref_points,
                                                      const vec3<float>* points, unsigned int Np,
                                                      QueryArgs qargs, const NeighborList* nlist)
{
    if (nlist != NULL)
    {
        return std::make_shared<NeighborListNeighborIterator>(nlist);
    }
    else
    {
        return std::make_shared<NeighborQueryNeighborIterator>(ref_points, points, Np, qargs);
    }
}

}; }; // namespace freud::locality
