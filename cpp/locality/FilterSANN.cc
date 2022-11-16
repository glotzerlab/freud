#include "FilterSANN.h"
#include "NeighborComputeFunctional.h"

namespace freud { namespace locality {

void FilterSANN::compute(const NeighborQuery* nq, const vec3<float>* query_points,
                         unsigned int num_query_points, const NeighborList* nlist, QueryArgs qargs)
{
    // make the unfiltered neighborlist from the arguments
    m_unfiltered_nlist = std::make_shared<NeighborList>(
        std::move(makeDefaultNlist(nq, nlist, query_points, num_query_points, qargs)));

    // work with sorted nlist
    NeighborList sorted_nlist(*m_unfiltered_nlist);
    sorted_nlist.sort(true);
};

}; }; // namespace freud::locality
