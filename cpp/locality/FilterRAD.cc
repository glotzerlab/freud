#include "FilterRAD.h"
#include "NeighborBond.h"
#include "NeighborComputeFunctional.h"
#include "utils.h"
#include <tbb/enumerable_thread_specific.h>
#include <vector>

namespace freud { namespace locality {

void FilterSANN::compute(const NeighborQuery* nq, const vec3<float>* query_points,
                         unsigned int num_query_points, const NeighborList* nlist, const QueryArgs& qargs)
{
    // make the unfiltered neighborlist from the arguments
    m_unfiltered_nlist = std::make_shared<NeighborList>(
        std::move(makeDefaultNlist(nq, nlist, query_points, num_query_points, qargs)));

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
    util::forLoopWrapper(0, sorted_nlist.getNumQueryPoints(), [&](size_t begin, size_t end) {
        // grab thread-local vector
        BondVector::reference local_bonds(filtered_bonds.local());
        for (auto i = begin; i < end; i++)
        {
            const unsigned int num_unfiltered_neighbors = sorted_counts(i);
            const unsigned int first_idx = sorted_nlist.find_first_index(i);

            // sum for the three closest neighbors
            for (unsigned int j = 0; j < num_unfiltered_neighbors; j++)
            {
                for (unsigned int k = 0; k < j; k++)
                {
                    
                }
            }

        }
    });

};

}; }; // namespace freud::locality
