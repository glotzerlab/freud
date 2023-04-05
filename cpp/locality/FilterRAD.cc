// Copyright (c) 2010-2023 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "FilterRAD.h"
#include "NeighborBond.h"
#include "NeighborComputeFunctional.h"
#include "utils.h"
#include <tbb/enumerable_thread_specific.h>
#include <vector>

namespace freud { namespace locality {

void FilterRAD::compute(const NeighborQuery* nq, const vec3<float>* query_points,
                        unsigned int num_query_points, const NeighborList* nlist, const QueryArgs& qargs)
{
    // make the unfiltered neighborlist from the arguments
    m_unfiltered_nlist = std::make_shared<NeighborList>(
        std::move(makeDefaultNlist(nq, nlist, query_points, num_query_points, qargs)));

    // work with nlist sorted by distance
    NeighborList sorted_nlist(*m_unfiltered_nlist);
    sorted_nlist.sort(true);

    // hold index of query point for a thread if its RAD shell isn't filled
    std::vector<unsigned int> unfilled_qps(sorted_nlist.getNumQueryPoints(),
                                           std::numeric_limits<unsigned int>::max());

    const auto& points = nq->getPoints();
    const auto& box = nq->getBox();
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
            const auto num_unfiltered_neighbors = sorted_counts(i);
            const auto first_idx = sorted_nlist.find_first_index(i);
            bool good_neighbor = true;

            // loop over each potential neighbor particle j
            for (unsigned int j = 0; j < num_unfiltered_neighbors; j++)
            {
                const auto first_neighbor_idx = sorted_neighbors(first_idx + j, 1);
                const auto v1 = box.wrap(query_points[i] - points[first_neighbor_idx]);
                good_neighbor = true;

                // loop over particles which may be blocking the neighbor j
                for (unsigned int k = 0; k < j; k++)
                {
                    const auto second_neighbor_idx = sorted_neighbors(first_idx + k, 1);
                    const auto v2 = box.wrap(query_points[i] - points[second_neighbor_idx]);

                    // check if k blocks j
                    if ((dot(v2, v2) * sorted_dist(first_idx + j) * sorted_dist(first_idx + k))
                        < (dot(v1, v2) * dot(v1, v1)))
                    {
                        good_neighbor = false;
                        break;
                    }
                }

                // if no k blocks j, add a bond from i to j
                if (good_neighbor)
                {
                    local_bonds.emplace_back(i, first_neighbor_idx, sorted_dist(first_idx + j));
                }
                else if (m_terminate_after_blocked)
                {
                    // if a particle blocks j and "RAD-closed" is requested
                    // stop looking for more neighbors
                    break;
                }
            }

            // if we have searched over all potential neighbors j and still have
            // not found one that is blocked, the neighbor shell may be incomplete.
            // This only applies to the RAD-closed case, because RAD-open will
            // never terminate prematurely.
            if (good_neighbor && m_terminate_after_blocked)
            {
                // in principle, the incomplete shell exception can be raised here,
                // but the error is more informative if the exception raised
                // includes each query point with an unfilled neighbor shell
                unfilled_qps[i] = i;
            }
        }
    });

    // print warning/exception about query point indices with unfilled neighbor shells
    Filter::warnAboutUnfilledNeighborShells(unfilled_qps);

    // combine thread-local arrays
    tbb::flattened2d<BondVector> flat_filtered_bonds = tbb::flatten2d(filtered_bonds);
    std::vector<NeighborBond> rad_bonds(flat_filtered_bonds.begin(), flat_filtered_bonds.end());

    // sort final bonds array by distance
    tbb::parallel_sort(rad_bonds.begin(), rad_bonds.end(), compareNeighborDistance);

    m_filtered_nlist = std::make_shared<NeighborList>(rad_bonds);
};

}; }; // namespace freud::locality
