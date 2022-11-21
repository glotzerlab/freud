#include "FilterSANN.h"
#include "NeighborComputeFunctional.h"
#include <vector>
#include "NeighborBond.h"

namespace freud { namespace locality {

void FilterSANN::compute(const NeighborQuery* nq, const vec3<float>* query_points,
                         unsigned int num_query_points, const NeighborList* nlist, QueryArgs qargs)
{
    // make the unfiltered neighborlist from the arguments
    // test if this copies the neighbor list.
    m_unfiltered_nlist = std::make_shared<NeighborList>(
        makeDefaultNlist(nq, nlist, query_points, num_query_points, qargs));
    
    // work with sorted nlist
    NeighborList sorted_nlist(*m_unfiltered_nlist);
    sorted_nlist.sort(true);
    
    auto sorted_neighbors=sorted_nlist.getNeighbors();
    auto sorted_dist=sorted_nlist.getDistances();
    auto sorted_weights=sorted_nlist.getWeights();
    auto sorted_counts = sorted_nlist.getCounts();

    std::vector<NeighborBond> filtered_bonds;

    // use the paralel for wrapper to loop over QP
    for (unsigned int i = 0; i< sorted_nlist.getNumQueryPoints();i++)
    {
        unsigned int m=3;
        unsigned int first_idx = sorted_nlist.find_first_index(i);
        float sum=0.0;
        // sum for the first three closest neighbors
        for (unsigned int j = 0; j < m && j < sorted_counts(i); j++)
        {
            sum += sorted_dist(first_idx + j);
            filtered_bonds.push_back(NeighborBond(i, sorted_neighbors(1, first_idx + j), sorted_dist(first_idx + j), sorted_weights(first_idx + j)));
        }
        while ((sum / (float(m) - 2.0)) > sorted_dist(first_idx + m) && m<sorted_counts(i))
        {
            sum += sorted_dist(first_idx + m);
            filtered_bonds.push_back(NeighborBond(i, sorted_neighbors(1, first_idx + m),
                                             sorted_dist(first_idx + m), sorted_weights(first_idx + 2)));
            m += 1;
        }

    }

    m_filtered_nlist = std::make_shared<NeighborList>(filtered_bonds);
};

}; }; // namespace freud::locality
