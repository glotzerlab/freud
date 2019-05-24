#ifndef NEIGHBOR_COMPUTE_FUNCTIONAL_H
#define NEIGHBOR_COMPUTE_FUNCTIONAL_H


#include "NeighborQuery.h"
// #include "AABBQuery.h"
#include "NeighborList.h"
// #include "Index1D.h"

namespace freud { namespace locality {


template<typename ComputePairType>
void loop_over_NeighborList(const NeighborList* nlist, ComputePairType cf)
    {
    const size_t *neighbor_list(nlist->getNeighbors());
    for (size_t bond = 0; bond < nlist->getNumBonds(); ++bond)
        {
        size_t i(neighbor_list[2*bond]);
        size_t j(neighbor_list[2*bond + 1]);
        cf(i, j);
        }
    }

template<typename ComputePairType>
void loop_over_NeighborList_parallel(const NeighborList* nlist, ComputePairType cf)
    {
    const size_t *neighbor_list(nlist->getNeighbors());
    size_t n_bonds = nlist->getNumBonds();
    parallel_for(tbb::blocked_range<size_t>(0, n_bonds),
        [=] (const tbb::blocked_range<size_t>& r)
        {
            for(size_t bond = r.begin(); bond !=r.end(); ++bond)
                {
                size_t i(neighbor_list[2*bond]);
                size_t j(neighbor_list[2*bond + 1]);
                cf(i, j);
                }
        });
    }


}; }; // end namespacec freud::locality

#endif