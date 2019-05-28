#ifndef NEIGHBOR_COMPUTE_FUNCTIONAL_H
#define NEIGHBOR_COMPUTE_FUNCTIONAL_H


#include "NeighborQuery.h"
// #include "AABBQuery.h"
#include "NeighborList.h"
// #include "Index1D.h"
#include <tbb/tbb.h>

namespace freud { namespace locality {


template<typename ComputePairType>
void loop_over_NeighborList(const NeighborList* nlist, const ComputePairType& cf)
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
void loop_over_NeighborList_parallel(const NeighborList* nlist, const ComputePairType& cf)
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

// Body should be an object taking in 
// with operator(size_t begin, size_t end)
template<typename Body>
void for_loop_wrapper(bool parallel, size_t begin, size_t end, const Body& body)
    {
        if(parallel)
            {
                tbb::parallel_for(tbb::blocked_range<size_t>(begin, end), 
                    [&body] (const tbb::blocked_range<size_t>& r) {body(r.begin(), r.end());});
            }
        else
            {
                body(begin, end);
            }
    }

}; }; // end namespace freud::locality

#endif