#ifndef NEIGHBOR_COMPUTE_FUNCTIONAL_H
#define NEIGHBOR_COMPUTE_FUNCTIONAL_H

#include <memory>
#include <tbb/tbb.h>

#include "AABBQuery.h"
#include "Index1D.h"
#include "NeighborList.h"
#include "NeighborQuery.h"

namespace freud { namespace locality {

// Body should be an object taking in
// with operator(size_t begin, size_t end)
template<typename Body> void for_loop_wrapper(bool parallel, size_t begin, size_t end, const Body& body)
{
    if (parallel)
    {
        tbb::parallel_for(tbb::blocked_range<size_t>(begin, end),
                          [&body](const tbb::blocked_range<size_t>& r) { body(r.begin(), r.end()); });
    }
    else
    {
        body(begin, end);
    }
}

// ComputePairType should be a void function that takes (ref_point, point) indices as input.
template<typename ComputePairType>
void loop_over_NeighborList(const NeighborQuery* ref_points, const vec3<float>* points, unsigned int Np,
                            QueryArgs qargs, const NeighborList* nlist, const ComputePairType& cf)
{
    // check if nlist exists
    if (nlist != NULL)
    {
        // if nlist exists, loop over it in parallel.
        loop_over_NeighborList_parallel(nlist, cf);
    }
    else
    {
        // if nlist does not exist, check if ref_points is an actual NeighborQuery
        std::shared_ptr<NeighborQueryIterator> iter;
        std::shared_ptr<AABBQuery> abq;
        if (const RawPoints* rp = dynamic_cast<const RawPoints*>(ref_points))
        {
            // if ref_points is RawPoints, build a NeighborQuery
            abq = std::make_shared<AABBQuery>(ref_points->getBox(), ref_points->getRefPoints(),
                                              ref_points->getNRef());
            iter = abq.get()->queryWithArgs(points, Np, qargs);
        }
        else
        {
            iter = ref_points->queryWithArgs(points, Np, qargs);
        }

        // iterate over the query object in parallel
        for_loop_wrapper(true, 0, Np, [iter, qargs, &cf](size_t begin, size_t end) {
            NeighborPoint np;
            for (size_t i = begin; i != end; ++i)
            {
                std::shared_ptr<NeighborQueryIterator> it = iter->query(i);
                np = it->next();
                while (!it->end())
                {
                    if (!qargs.exclude_ii || i != np.ref_id)
                    {
                        cf(np.ref_id, i);
                    }
                    np = it->next();
                }
            }
        });
    }
}

// ComputePairType should be a void function that takes (ref_point, point) indices as input.
template<typename ComputePairType>
void loop_over_NeighborList_parallel(const NeighborList* nlist, const ComputePairType& cf)
{
    const size_t* neighbor_list(nlist->getNeighbors());
    size_t n_bonds = nlist->getNumBonds();
    parallel_for(tbb::blocked_range<size_t>(0, n_bonds), [=](const tbb::blocked_range<size_t>& r) {
        for (size_t bond = r.begin(); bond != r.end(); ++bond)
        {
            size_t i(neighbor_list[2 * bond]);
            size_t j(neighbor_list[2 * bond + 1]);
            cf(i, j);
        }
    });
}

}; }; // end namespace freud::locality

#endif
