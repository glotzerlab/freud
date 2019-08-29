#ifndef NEIGHBOR_COMPUTE_FUNCTIONAL_H
#define NEIGHBOR_COMPUTE_FUNCTIONAL_H

#include <memory>
#include <tbb/tbb.h>

#include "AABBQuery.h"
#include "Index1D.h"
#include "NeighborList.h"
#include "NeighborQuery.h"

/*! \file NeighborComputeFunctional.h
    \brief Implements logic for looping over NeighborQuery
*/

namespace freud { namespace locality {

class NeighborListPerPointIterator : public NeighborPerPointIterator
{
public:
    NeighborListPerPointIterator(const NeighborList* nlist, size_t point_index):
        NeighborPerPointIterator(point_index), m_nlist(nlist)
        {
            m_current_index = m_nlist->find_first_index(point_index);
            m_returned_point_index = m_nlist->getNeighbors()[2 * m_current_index];
            at_end = m_current_index == m_nlist->getNumBonds();
        } 
    
    ~NeighborListPerPointIterator() {}

    virtual NeighborBond next()
    {
        if(m_current_index == m_nlist->getNumBonds())
        {
            at_end = true;
            return NeighborBond();
        }

        NeighborBond nb = NeighborBond(m_nlist->getNeighbors()[2 * m_current_index],
                                        m_nlist->getNeighbors()[2 * m_current_index + 1], 
                                        m_nlist->getDistances()[m_current_index],
                                        m_nlist->getWeights()[m_current_index]);
        ++m_current_index;
        m_returned_point_index = nb.id;
        return nb;
    }

    virtual bool end()
    {
        return (m_returned_point_index != m_query_point_idx) || at_end;
    }

private:
    const NeighborList* m_nlist;
    size_t m_current_index;
    size_t m_returned_point_index;
    bool at_end;
};


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


//! Wrapper iterating looping over NeighborQuery or NeighborList
/*! \param neighbor_query NeighborQuery object to iterate over
    \param query_points Points
    \param n_query_points Number of query_points
    \param qargs Query arguments
    \param nlist Neighbor List. If not NULL, loop over it. Otherwise, use neighbor_query
           appropriately with given qargs.
    \param cf An object with
           operator(size_t point_index, std::shared_ptr<NeighborIterator>) as input.
           It should implement iteration logic over the iterator.
*/
template<typename ComputePairType>
void loopOverNeighborsIterator(const NeighborQuery* neighbor_query, const vec3<float>* query_points, unsigned int n_query_points,
                            QueryArgs qargs, const NeighborList* nlist, 
                            const ComputePairType& cf, bool parallel = true)
{
    // check if nlist exists
    if (nlist != NULL)
    {
        forLoopWrapper(0, n_query_points, [=](size_t begin, size_t end) {
            for (size_t i = begin; i != end; ++i)
            {
                std::shared_ptr<NeighborListPerPointIterator> niter = std::make_shared<NeighborListPerPointIterator>(nlist, i);
                cf(i, niter);
            }
        }, parallel);
    }
    else
    {
        // if nlist does not exist, check if neighbor_query is an actual NeighborQuery
        std::shared_ptr<NeighborQueryIterator> iter;
        std::shared_ptr<AABBQuery> abq;
        // check if neighbor_query is a pointer to a RawPoints object
        // dynamic_cast will fail if neighbor_query is not actually pointing to RawPoints
        // and return a null pointer. Then, the assignment operator will return
        // a null pointer, making the condition in the if statement to be false.
        // This is a typical C++ way of checking the type of a polymorphic class
        // using pointers and casting.
        if (dynamic_cast<const RawPoints*>(neighbor_query))
        {
            // if neighbor_query is RawPoints, build a NeighborQuery
            abq = std::make_shared<AABBQuery>(neighbor_query->getBox(), neighbor_query->getPoints(),
                                            neighbor_query->getNPoints());
            iter = abq->queryWithArgs(query_points, n_query_points, qargs);
        }
        else
        {
            iter = neighbor_query->queryWithArgs(query_points, n_query_points, qargs);
        }

        // iterate over the query object in parallel
        forLoopWrapper(0, n_query_points, [=](size_t begin, size_t end) {
            NeighborBond nb;
            for (size_t i = begin; i != end; ++i)
            {
                std::shared_ptr<NeighborQueryPerPointIterator> it = iter->query(i);
                cf(i, it);
            }
        }, parallel);
    }
}


//! Wrapper iterating looping over NeighborQuery or NeighborList
/*! \param neighbor_query NeighborQuery object to iterate over
    \param query_points Points
    \param n_query_points Number of query_points
    \param qargs Query arguments
    \param nlist Neighbor List. If not NULL, loop over it. Otherwise, use neighbor_query
           appropriately with given qargs.
    \param cf An object with operator(NeighborBond) as input.
*/
template<typename ComputePairType>
void loopOverNeighbors(const NeighborQuery* neighbor_query, const vec3<float>* query_points, unsigned int n_query_points,
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
        loopOverNeighborQuery(neighbor_query, query_points, n_query_points, qargs, cf, parallel);
    }
}

//! Wrapper looping over NeighborList in parallel.
/*! \param nlist Neighbor List to loop over.
    \param cf An object with operator(NeighborBond) as input.
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
            size_t point_index(neighbor_list[2 * bond]);
            size_t ref_point_index(neighbor_list[2 * bond + 1]);
            const NeighborBond nb(point_index, ref_point_index, neighbor_distances[bond], neighbor_weights[bond]);
            cf(nb);
        }
    }, parallel);
}

//! Wrapper looping over NeighborQuery in parallel
/*! \param neighbor_query NeighborQuery object to iterate over
    \param query_points Points
    \param n_query_points Number of query_points
    \param qargs Query arguments
    \param cf An object with operator(NeighborBond) as input.
*/
template<typename ComputePairType>
void loopOverNeighborQuery(const NeighborQuery* neighbor_query, const vec3<float>* query_points,
                           unsigned int n_query_points, QueryArgs qargs, const ComputePairType& cf, bool parallel)
{
    // if nlist does not exist, check if neighbor_query is an actual NeighborQuery
    std::shared_ptr<NeighborQueryIterator> iter;
    std::shared_ptr<AABBQuery> abq;
    // check if neighbor_query is a pointer to a RawPoints object
    // dynamic_cast will fail if neighbor_query is not actually pointing to RawPoints
    // and return a null pointer. Then, the assignment operator will return
    // a null pointer, making the condition in the if statement to be false.
    // This is a typical C++ way of checking the type of a polymorphic class
    // using pointers and casting.
    if (dynamic_cast<const RawPoints*>(neighbor_query))
    {
        // if neighbor_query is RawPoints, build a NeighborQuery
        abq = std::make_shared<AABBQuery>(neighbor_query->getBox(), neighbor_query->getPoints(),
                                          neighbor_query->getNPoints());
        iter = abq->queryWithArgs(query_points, n_query_points, qargs);
    }
    else
    {
        iter = neighbor_query->queryWithArgs(query_points, n_query_points, qargs);
    }

    // iterate over the query object in parallel
    forLoopWrapper(0, n_query_points, [&iter, &qargs, &cf](size_t begin, size_t end) {
        NeighborBond nb;
        for (size_t i = begin; i != end; ++i)
        {
            std::shared_ptr<NeighborQueryPerPointIterator> it = iter->query(i);
            nb = it->next();
            while (!it->end())
            {
                cf(nb);
                nb = it->next();
            }
        }
    }, parallel);
}

}; }; // end namespace freud::locality

#endif
