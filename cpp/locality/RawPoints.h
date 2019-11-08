// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef RAW_POINTS_H
#define RAW_POINTS_H

#include <stdexcept>

#include "AABBQuery.h"
#include "NeighborQuery.h"

/*! \file RawPoints.h
    \brief Defines a simplest NeighborQuery object that actually farms out
           querying logic to an AABBQuery.
*/

namespace freud { namespace locality {

//! Dummy class to just contain minimal information and not actually query.
/*! The purpose of this class is to support dynamic NeighborQuery object
 *  resolution. Users may pass instances of this class instead of providing a
 *  NeighborQuery to various compute functions throughout freud, which is an
 *  indication that the function needs to compute its own NeighborQuery. That
 *  logic, which is primary encapsulated in the NeighborComputeFunctional.h
 *  file, helps provide a nice Python API as well.
 */
class RawPoints : public NeighborQuery
{
public:
    RawPoints() {}

    RawPoints(const box::Box& box, const vec3<float>* points, unsigned int n_points)
        : NeighborQuery(box, points, n_points)
    {}

    ~RawPoints() {}

    //! Perform a query based on a set of query parameters.
    /*! Shadow parent function to ensure that the underlying AABBQuery is
     * only constructed when this object is actually queried. Note that unlike
     * the parent function it is not const since it does modify the object.
     *
     *  \param query_points The points to find neighbors for.
     *  \param n_query_points The number of query points.
     *  \param qargs The query arguments that should be used to find neighbors.
     */
    virtual std::shared_ptr<NeighborQueryIterator>
    query(const vec3<float>* query_points, unsigned int n_query_points, QueryArgs query_args) const
    {
        if (!aq)
        {
            aq = std::unique_ptr<AABBQuery>(new AABBQuery(m_box, m_points, m_n_points));
        }

        this->validateQueryArgs(query_args);
        return std::make_shared<NeighborQueryIterator>(this, query_points, n_query_points, query_args);
    }

    // dummy implementation for pure virtual function in the parent class
    virtual std::shared_ptr<NeighborQueryPerPointIterator>
    querySingle(const vec3<float> query_point, unsigned int query_point_idx, QueryArgs qargs) const
    {
        if (!aq)
        {
            throw std::runtime_error("The underlying AABBQuery object has not yet been initialized. Please "
                                     "report this error.");
        }

        return aq->querySingle(query_point, query_point_idx, qargs);
    }

private:
    mutable std::unique_ptr<AABBQuery> aq; //!< The AABBQuery object that will be used to perform queries.
};

}; }; // end namespace freud::locality

#endif // NEIGHBOR_QUERY_H
