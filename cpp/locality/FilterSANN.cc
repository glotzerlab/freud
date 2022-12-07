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
    m_unfiltered_nlist = std::make_shared<NeighborList>(
        std::move(makeDefaultNlist(nq, nlist, query_points, num_query_points, qargs))
    );

    // work with nlist sorted by distance
    NeighborList sorted_nlist(*m_unfiltered_nlist);
    sorted_nlist.sort(true);

    const auto& sorted_neighbors = sorted_nlist.getNeighbors();
    const auto& sorted_dist = sorted_nlist.getDistances();
    const auto& sorted_weights = sorted_nlist.getWeights();
    const auto& sorted_counts = sorted_nlist.getCounts();

    using BondVector = tbb::enumerable_thread_specific<std::vector<NeighborBond>>;
    BondVector filtered_bonds;

    // parallelize over query_point_index
    util::forLoopWrapper(0, sorted_nlist.getNumQueryPoints(), [&](size_t begin, size_t end)
    {
        // grab thread-local vector
        BondVector::reference local_bonds(filtered_bonds.local());
        for (auto i = begin; i < end; i++)
        {
            unsigned int m = 3;
            const unsigned int num_unfiltered_neighbors = sorted_counts(i);
            const unsigned int first_idx = sorted_nlist.find_first_index(i);
            float sum = 0.0;

            // sum for the three closest neighbors
            for (unsigned int j = 0; j < m && j < num_unfiltered_neighbors; j++)
            {
                const unsigned int neighbor_idx = first_idx + j;
                sum += sorted_dist(neighbor_idx);
                local_bonds.emplace_back(i, sorted_neighbors(neighbor_idx, 1),
                                            sorted_dist(neighbor_idx),
                                            sorted_weights(neighbor_idx));
            }

            // add neighbors after adding the first three
            while (m < num_unfiltered_neighbors && (sum / (float(m) - 2.0)) > sorted_dist(first_idx + m))
            {
                const unsigned int neighbor_idx = first_idx + m;
                sum += sorted_dist(neighbor_idx);
                local_bonds.emplace_back(i, sorted_neighbors(neighbor_idx, 1),
                                            sorted_dist(neighbor_idx),
                                            sorted_weights(neighbor_idx));
                m += 1;
            }
        }
    });

    // combine thread-local NeighborBond vectors into a single vector
    tbb::flattened2d<BondVector> flat_filtered_bonds = tbb::flatten2d(filtered_bonds);
    std::vector<NeighborBond> sann_bonds(flat_filtered_bonds.begin(), flat_filtered_bonds.end());

    // sort final bonds array by distance
    tbb::parallel_sort(sann_bonds.begin(), sann_bonds.end(), compareNeighborDistance);
    //tbb::parallel_sort(sann_bonds.begin(), sann_bonds.end(), compareNeighborBond);

    m_filtered_nlist = std::make_shared<NeighborList>(sann_bonds);
};

}; }; // namespace freud::locality
