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

//! Abstract class for generating appropriate iterator of neighbors per point.
class NeighborIterator 
{
public:
    NeighborIterator () {}
    virtual ~NeighborIterator() {}

    //! Abstract class for iterator of neighbors per point.
    class PerPointIterator
    {
    public:
        PerPointIterator(size_t point_index): m_point_index(point_index) {}
        virtual ~PerPointIterator() {}

        //! Get the next NeighborBond
        virtual NeighborBond next() = 0;

        //! End of iteration
        virtual bool end() const = 0;

    protected:
        size_t m_point_index;
    };

    // Get the iterator of the neighbors of given point_index.
    virtual std::shared_ptr<PerPointIterator> queryPerPoint(size_t point_index) = 0;
};

//! NeighborIterator for NeighborList
class NeighborListNeighborIterator : public NeighborIterator
{
public:
    NeighborListNeighborIterator(const NeighborList * nlist):
        m_nlist(nlist) {}

    ~NeighborListNeighborIterator() {}

    //! PerPointIterator for NeighborList
    class NeighborListPerPointIterator : public PerPointIterator
    {
    public:
        NeighborListPerPointIterator(const NeighborList* nlist, size_t point_index):
            PerPointIterator(point_index), m_nlist(nlist)
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

        virtual bool end() const
        {
            return (m_returned_point_index != m_point_index) || at_end;
        }

    private:
        const NeighborList* m_nlist;
        size_t m_current_index;
        size_t m_returned_point_index;
        bool at_end;
    };

    virtual std::shared_ptr<PerPointIterator> queryPerPoint(size_t point_index)
    {
        return std::make_shared<NeighborListPerPointIterator>(m_nlist, point_index);
    }

private:
    const NeighborList * m_nlist;
};

//! NeighborIterator for NeighborQuery
class NeighborQueryNeighborIterator : public NeighborIterator
{
public:
    NeighborQueryNeighborIterator(const NeighborQuery* nq, const vec3<float> *points, unsigned int N, QueryArgs qargs)
    {
        m_qargs = qargs;
        // The query iterators always finds the point with itself as a neighbor bond.
        // To find the proper number of neighbors, we need to increment
        // the number of neighbors to look for.
        if(qargs.exclude_ii && (qargs.mode == QueryArgs::QueryType::nearest))
        {
            ++m_qargs.nn;
        }

        // check if nq is a pointer to a RawPoints object
        // dynamic_cast will fail if nq is not actually pointing to RawPoints
        // and return a null pointer. Then, the assignment operator will return
        // a null pointer, making the condition in the if statement to be false.
        // This is a typical C++ way of checking the type of a polymorphic class
        // using pointers and casting.
        if (dynamic_cast<const RawPoints*>(nq))
        {
            // if nq is RawPoints, build a NeighborQuery
            m_abq = std::make_shared<AABBQuery>(nq->getBox(), nq->getPoints(),
                                              nq->getNPoints());
            m_nqiter = m_abq->queryWithArgs(points, N, m_qargs);
        }
        else
        {
            m_nqiter = nq->queryWithArgs(points, N, m_qargs);
        }
    }

    ~NeighborQueryNeighborIterator() {}

    //! PerPointIterator for NeighborQuery
    class NeighborQueryPerPointIterator : public PerPointIterator
    {
    public:
        NeighborQueryPerPointIterator(std::shared_ptr<NeighborQueryIterator> nqiter, size_t point_index, bool exclude_ii):
        PerPointIterator(point_index), m_nqiter(nqiter), m_exclude_ii(exclude_ii) {}

        ~NeighborQueryPerPointIterator() {}

        virtual NeighborBond next()
        {
            NeighborBond nb = m_nqiter->next();
            nb.id = m_point_index;
            if (!m_exclude_ii || m_point_index != nb.ref_id)
            {
                return nb;
            }
            else 
            {
                nb = m_nqiter->next();
                nb.id = m_point_index;
                return nb;
            }
        }

        virtual bool end() const
        {
            return m_nqiter->end();
        }
    private:
        std::shared_ptr<NeighborQueryIterator> m_nqiter;
        bool m_exclude_ii;
    };

    virtual std::shared_ptr<PerPointIterator> queryPerPoint(size_t point_index)
    {
        std::shared_ptr<NeighborQueryIterator> iter = m_nqiter->query(point_index);
        return std::make_shared<NeighborQueryPerPointIterator>(iter, point_index, m_qargs.exclude_ii);
    }

private:
    std::shared_ptr<AABBQuery> m_abq;
    std::shared_ptr<NeighborQueryIterator> m_nqiter;
    QueryArgs m_qargs;
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

//! Appropriately generate NeighborIterator
/*! \param neighbor_query NeighborQuery object to iterate over
    \param query_points Points
    \param n_query_points Number of query_points
    \param qargs Query arguments
    \param nlist Neighbor List. If not NULL, generate an iterator based on it.
        Otherwise, use neighbor_query appropriately with given qargs.
*/
std::shared_ptr<NeighborIterator> getNeighborIterator(
    const NeighborQuery* neighbor_query, const vec3<float>* query_points, unsigned int n_query_points,
    QueryArgs qargs, const NeighborList* nlist);

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
    std::shared_ptr<NeighborIterator> niter = getNeighborIterator(neighbor_query, query_points, n_query_points, qargs, nlist);
    forLoopWrapper(0, n_query_points, [=](size_t begin, size_t end) {
        for (size_t i = begin; i != end; ++i)
        {
            auto ppiter = niter->queryPerPoint(i);
            cf(i, ppiter);
        }
    }, parallel);
}

//! Wrapper iterating looping over NeighborQuery or NeighborList
/*! \param neighbor_query NeighborQuery object to iterate over
    \param query_points Points
    \param n_query_points Number of query_points
    \param qargs Query arguments
    \param nlist Neighbor List. If not NULL, loop over it. Otherwise, use neighbor_query
           appropriately with given qargs.
    \param cf An object with
           operator(size_t ref_point_index, size_t point_index,
               float distance, float weight) as input.
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
            size_t point_index(neighbor_list[2 * bond]);
            size_t ref_point_index(neighbor_list[2 * bond + 1]);
            cf(ref_point_index, point_index, neighbor_distances[bond], neighbor_weights[bond]);
        }
    }, parallel);
}

//! Wrapper looping over NeighborQuery in parallel
/*! \param neighbor_query NeighborQuery object to iterate over
    \param query_points Points
    \param n_query_points Number of query_points
    \param qargs Query arguments
    \param cf An object with
           operator(size_t ref_point_index, size_t point_index,
               float distance, float weight) as input.
*/
template<typename ComputePairType>
void loopOverNeighborQuery(const NeighborQuery* neighbor_query, const vec3<float>* query_points,
                           unsigned int n_query_points, QueryArgs qargs, const ComputePairType& cf, bool parallel)
{
    // The query iterators always finds the point with itself as a neighbor bond.
    // To find the proper number of neighbors, we need to increment
    // the number of neighbors to look for.
    if(qargs.exclude_ii && (qargs.mode == QueryArgs::QueryType::nearest))
    {
        ++qargs.nn;
    }
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
        NeighborBond np;
        for (size_t i = begin; i != end; ++i)
        {
            std::shared_ptr<NeighborQueryIterator> it = iter->query(i);
            np = it->next();
            while (!it->end())
            {
                //! Warning! If qargs.exclude_ii is true, NeighborBond with same indices
                // will not be considered regardless of neighbor_query and query_points
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

}; }; // end namespace freud::locality

#endif
