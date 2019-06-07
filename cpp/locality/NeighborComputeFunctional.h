#ifndef NEIGHBOR_COMPUTE_FUNCTIONAL_H
#define NEIGHBOR_COMPUTE_FUNCTIONAL_H

#include <memory>
#include <tbb/tbb.h>

#include "NeighborQuery.h"
#include "AABBQuery.h"
#include "NeighborList.h"
#include "Index1D.h"

namespace freud { namespace locality {

class RawPoints : public NeighborQuery
    {
    public:
        RawPoints();

        RawPoints(const box::Box &box, const vec3<float> *ref_points, unsigned int Nref) :
            NeighborQuery(box, ref_points, Nref) {}

        ~RawPoints() {}

        // dummy implementation for pure virtual function in the parent class
        virtual std::shared_ptr<NeighborQueryIterator> query(const vec3<float> *points, unsigned int N, unsigned int k, bool exclude_ii=false) const
        {
            return nullptr;
        }

        // dummy implementation for pure virtual function in the parent class
        virtual std::shared_ptr<NeighborQueryIterator> queryBall(const vec3<float> *points, unsigned int N, float r, bool exclude_ii=false) const
        {
            return nullptr;
        }

    };

// ComputePairType should be a void function that takes (ref_point, point) as input.
template<typename ComputePairType>
void loop_over_NeighborList(const NeighborQuery *ref_points, const vec3<float> *points, unsigned int Np,
                                  QueryArgs qargs, const NeighborList* nlist, const ComputePairType& cf)
    {
    // check if nlist exists
    if(nlist != NULL)
        {
        // if nlist exists, loop over it parallely
            loop_over_NeighborList_parallel(nlist, cf);
        }
    else
        {
        // if nlist does not exist, check if ref_points is an actual NeighborQuery
        std::shared_ptr<NeighborQueryIterator> iter;
        std::shared_ptr<AABBQuery> abq;
        if(const RawPoints* rp = dynamic_cast<const RawPoints*>(ref_points))
            {
            // if ref_points is RawPoints, build a NeighborQuery
            abq = std::make_shared<AABBQuery>(ref_points->getBox(), ref_points->getRefPoints(), ref_points->getNRef());
            iter = abq.get()->queryWithArgs(points, Np, qargs);
            }
        else
            {
            iter = ref_points->queryWithArgs(points, Np, qargs);
            }

        // iterate over the query object parallely
        tbb::parallel_for(tbb::blocked_range<size_t>(0, Np),
            [&] (const tbb::blocked_range<size_t> &r)
            {
            NeighborPoint np;
            for (size_t i(r.begin()); i != r.end(); ++i)
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

// ComputePairType should be a void function that takes (ref_point, point) as input.
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
