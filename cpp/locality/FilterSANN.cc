#include "FilterSANN.h"
#include "NeighborComputeFunctional.h"


void FilterSANN::compute(const NeighborQuery *nq, const vec3<float> *query_points,
                unsigned int num_query_points, const NeighborList *nlist,
                QueryArgs qargs) override
{
    // make the unfiltered neighborlist from the arguments
    m_unfiltered_nlist = makeDefaultNlist(nq, nlist, query_points, num_query_points, qargs);

    // do stuff with the unfiltered neighborlist to make the filtered neighborlist
};
