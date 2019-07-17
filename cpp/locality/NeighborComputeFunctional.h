#ifndef NEIGHBOR_COMPUTE_FUNCTIONAL_H
#define NEIGHBOR_COMPUTE_FUNCTIONAL_H

#include <memory>
#include <tbb/tbb.h>

#include "AABBQuery.h"
#include "Index1D.h"
#include "NeighborList.h"
#include "NeighborQuery.h"

namespace freud { namespace locality {
//! Wrapper for for-loop to allow the execution in parallel or not.
/*! \param parallel If true, run body in parallel.
    \param begin Beginning index.
    \param end Ending index.
    \param body An object
           with operator(size_t begin, size_t end).
*/
template<typename Body> void forLoopWrapper(size_t begin, size_t end, const Body& body, bool parallel)
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

//! Wrapper iterating over NeighborQuery or NeighborList
/*! \param ref_points NeighborQuery object to iterate over
    \param points Points
    \param Np Number of points
    \param qargs Query arguments
    \param nlist Neighbor List. If not NULL, loop over it. Otherwise, use ref_points
           appropriately with given qargs.
    \param cf An object with
           operator(size_t ref_point_index, size_t point_index,
               float distance, float weight) as input.
*/
template<typename ComputePairType>
void loopOverNeighbors(const NeighborQuery* ref_points, const vec3<float>* points, unsigned int Np,
                       QueryArgs qargs, const NeighborList* nlist, const ComputePairType& cf,
                       bool parallel = true)
{
    // check if nlist exists
    if (nlist != NULL)
    {
        // if nlist exists, loop over it in parallel.
        loopOverNeighborList(nlist, cf, parallel);
    }
    else
    {
        loopOverNeighborQuery(ref_points, points, Np, qargs, cf, parallel);
    }
}

//! Wrapper looping over NeighborList in parallel.
/*! \param nlist Neighbor List to loop over.
    \param cf An object with
           operator(size_t ref_point_index, size_t point_index,
               float distance, float weight) as input.
*/
template<typename ComputePairType>
void loopOverNeighborList(const NeighborList* nlist, const ComputePairType& cf, bool parallel)
{
    const size_t* neighbor_list(nlist->getNeighbors());
    size_t n_bonds = nlist->getNumBonds();
    const float* neighbor_distances = nlist->getDistances();
    const float* neighbor_weights = nlist->getWeights();
    forLoopWrapper(0, n_bonds, [=](size_t begin, size_t end) {
        for (size_t bond = begin; bond != end; ++bond)
        {
            size_t i(neighbor_list[2 * bond]);
            size_t j(neighbor_list[2 * bond + 1]);
            cf(i, j, neighbor_distances[bond], neighbor_weights[bond]);
        }
    }, parallel);
}

//! Wrapper looping over NeighborQuery in parallel
/*! \param ref_points NeighborQuery object to iterate over
    \param points Points
    \param Np Number of points
    \param qargs Query arguments
    \param cf An object with
           operator(size_t ref_point_index, size_t point_index,
               float distance, float weight) as input.
*/
template<typename ComputePairType>
void loopOverNeighborQuery(const NeighborQuery* ref_points, const vec3<float>* points,
                           unsigned int Np, QueryArgs qargs, const ComputePairType& cf, bool parallel)
{
    // if nlist does not exist, check if ref_points is an actual NeighborQuery
    std::shared_ptr<NeighborQueryIterator> iter;
    std::shared_ptr<AABBQuery> abq;
    // check if ref_points is a pointer to a RawPoints object
    // dynamic_cast will fail if ref_points is not actually pointing to RawPoints
    // and return a null pointer. Then, the assignment operator will return
    // a null pointer, making the condition in the if statement to be false.
    // This is a typical C++ way of checking the type of a polymorphic class
    // using pointers and casting.
    if (const RawPoints* rp = dynamic_cast<const RawPoints*>(ref_points))
    {
        // if ref_points is RawPoints, build a NeighborQuery
        abq = std::make_shared<AABBQuery>(ref_points->getBox(), ref_points->getRefPoints(),
                                          ref_points->getNRef());
        iter = abq->queryWithArgs(points, Np, qargs);
    }
    else
    {
        iter = ref_points->queryWithArgs(points, Np, qargs);
    }

    // iterate over the query object in parallel
    forLoopWrapper(0, Np, [&iter, &qargs, &cf](size_t begin, size_t end) {
        NeighborBond np;
        for (size_t i = begin; i != end; ++i)
        {
            std::shared_ptr<NeighborQueryIterator> it = iter->query(i);
            np = it->next();
            while (!it->end())
            {
                //! Warning! If qargs.exclude_ii is true, NeighborBond with same indices
                // will not be considered regardless of ref_points and points
                // being same set of points
                if (!qargs.exclude_ii || i != np.ref_id)
                {
                    cf(np.ref_id, i, np.distance, np.weight);
                }
                np = it->next();
            }
        }
    }, parallel);
}

// This function does not work for now since ref_point point orders are different
// for NeighborList and NeighborQuery.query().
//! Wrapper looping over NeighborQuery or NeighborList
/*! \param ref_points NeighborQuery object to iterate over
    \param points Points
    \param Np Number of points
    \param qargs Query arguments
    \param nlist Neighbor List. If not NULL, loop over it. Otherwise, use ref_points
           appropriately with given qargs.
    \param cf A void function that takes
           (ref_point_index, point_index, distance, weight) as input.
*/
template<typename ComputePairType, typename PreprocessType, typename PostprocessType>
void loopOverNeighborsPoint(const NeighborQuery* ref_points, const vec3<float>* points, unsigned int Np,
                            QueryArgs qargs, const NeighborList* nlist, const ComputePairType& cf,
                            const PreprocessType& pre, const PostprocessType& post, bool parallel = true)
{
    // check if nlist exists
    if (nlist != NULL)
    {
        // if nlist exists, loop over it in parallel.
        loopOverNeighborListPoint(nlist, Np, cf, pre, post, parallel);
    }
    else
    {
        loopOverNeighborQueryPoint(ref_points, points, Np, qargs, cf, pre, post, parallel);
    }
}

// This function does not work for now since ref_point point orders are different
// for NeighborList and NeighborQuery.query().
//! Wrapper looping over NeighborList per ref_point in parallel.
/*! \param nlist NeighborList to loop over.
    \param cf A void function that takes
           (ref_point_index, point_index, distance, weight) as input.
*/
template<typename ComputePairType, typename PreprocessType, typename PostprocessType>
void loopOverNeighborListPoint(const NeighborList* nlist, unsigned int Np, const ComputePairType& cf,
                               const PreprocessType& pre, const PostprocessType& post, bool parallel)
{
    const size_t* neighbor_list(nlist->getNeighbors());
    const float* neighbor_distances = nlist->getDistances();
    const float* neighbor_weights = nlist->getWeights();
    forLoopWrapper(0, Np, [=](size_t begin, size_t end) {
        size_t bond(nlist->find_first_index(begin));
        for (size_t i = begin; i != end; ++i)
        {
            auto data = pre(i);
            for (; bond < nlist->getNumBonds() && neighbor_list[2 * bond] == i; ++bond)
            {
                const size_t j(neighbor_list[2 * bond + 1]);
                cf(i, j, neighbor_distances[bond], neighbor_weights[bond], &data);
            }
            post(i, &data);
        }
    }, parallel);
}

// This function does not work for now since ref_point point orders are different
// for NeighborList and NeighborQuery.query().
//! Wrapper looping over NeighborQuery
/*! \param ref_points NeighborQuery object to iterate over
    \param points Points
    \param Np Number of points
    \param qargs Query arguments
    \param cf A void function that takes
           (ref_point_index, point_index, distance, weight) as input.
*/
template<typename ComputePairType, typename PreprocessType, typename PostprocessType>
void loopOverNeighborQueryPoint(const NeighborQuery* ref_points, const vec3<float>* points, unsigned int Np,
                                QueryArgs qargs, const ComputePairType& cf, const PreprocessType& pre,
                                const PostprocessType& post, bool parallel)
{
    // if nlist does not exist, check if ref_points is an actual NeighborQuery
    std::shared_ptr<NeighborQueryIterator> iter;
    std::shared_ptr<AABBQuery> abq;
    // check if ref_points is a pointer to a RawPoints object
    // dynamic_cast will fail if ref_points is not actually pointing to RawPoints
    // and return a null pointer. Then, the assignment operator will return
    // a null pointer, making the condition in the if statement to be false.
    // This is a typical C++ way of checking the type of a polymorphic class
    // using pointers and casting.
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
    forLoopWrapper(0, Np, [&iter, &qargs, &cf, &pre, &post](size_t begin, size_t end) {
        NeighborBond np;
        for (size_t i = begin; i != end; ++i)
        {
            std::shared_ptr<NeighborQueryIterator> it = iter->query(i);
            auto data = pre(i);
            np = it->next();
            while (!it->end())
            {
                //! Warning! If qargs.exclude_ii is true, NeighborBond with same indices
                // will not be considered regardless of ref_points and points
                // being same set of points
                if (!qargs.exclude_ii || i != np.ref_id)
                {
                    // TODO when Voronoi gets incorporated in NeighborQuery infrastructure
                    // weight set to 1 for now
                    cf(np.ref_id, i, np.distance, np.weight, &data);
                }
                np = it->next();
            }
            post(i, &data);
        }
    }, parallel);
}

}; }; // end namespace freud::locality

#endif
