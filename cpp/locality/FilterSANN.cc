#include "FilterSANN.h"
#include "NeighborComputeFunctional.h"
#include <vector>
#include "NeighborBond.h"
#include <tbb/enumerable_thread_specific.h>
#include "utils.h"

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

    //std::vector<NeighborBond> filtered_bonds;
    using BondVector = tbb::enumerable_thread_specific<std::vector<NeighborBond>>;
    BondVector filtered_bonds;

    // use the paralel for wrapper to loop over QP
    util::forLoopWrapper(0,sorted_nlist.getNumQueryPoints(), [&](size_t begin, size_t end)
    {
        // am i mising any other local assignments?
        BondVector::reference local_bonds(filtered_bonds.local());
        for (auto i = begin; i < end; i++)
        {
            unsigned int m=3;
            unsigned int first_idx = sorted_nlist.find_first_index(i);
            float sum=0.0;
            // sum for the first three closest neighbors
            for (unsigned int j = 0; j < m && j < sorted_counts(i); j++)
            {
                sum += sorted_dist(first_idx + j);
                local_bonds.emplace_back(i, sorted_neighbors(first_idx + j, 1), sorted_dist(first_idx + j), sorted_weights(first_idx + j));
            }
            while (m<sorted_counts(i) && (sum / (float(m) - 2.0)) > sorted_dist(first_idx + m))
            {
                sum += sorted_dist(first_idx + m);
                local_bonds.emplace_back(i, sorted_neighbors(first_idx + m, 1),
                                                 sorted_dist(first_idx + m), sorted_weights(first_idx + 2));
                m += 1;
            }

        }
    });

    tbb::flattened2d<BondVector> flat_filtered_bonds = tbb::flatten2d(filtered_bonds);
    std::vector<NeighborBond> sann_bonds(flat_filtered_bonds.begin(), flat_filtered_bonds.end());
    tbb::parallel_sort(sann_bonds.begin(), sann_bonds.end(), compareNeighborDistance);
    //tbb::parallel_sort(sann_bonds.begin(), sann_bonds.end(), compareNeighborBond);

    // sort by distances after paralel for loop

    m_filtered_nlist = std::make_shared<NeighborList>(sann_bonds);
};

}; }; // namespace freud::locality
