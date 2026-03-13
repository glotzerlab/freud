// Copyright (c) 2010-2026 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef RAW_POINTS_H
#define RAW_POINTS_H

#include <memory>
#include <stdexcept>

#include "AABBQuery.h"
#include "Box.h"
#include "LinearCell.h"
#include "NeighborQuery.h"
#include "VectorMath.h"

/*! \file RawPoints.h
    \brief Defines a simplest NeighborQuery object that actually farms out
           querying logic to a CellQuery.
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
    RawPoints() = default;

    RawPoints(const box::Box& box, const vec3<float>* points, unsigned int n_points)
        : NeighborQuery(box, points, n_points)
    {}

    ~RawPoints() override = default;

    //! Perform a query based on a set of query parameters.
    /*! Shadow parent function to ensure that the underlying CellQuery is
     * only constructed when this object is actually queried. Note that unlike
     * the parent function it is not const since it does modify the object.
     *
     *  \param query_points The points to find neighbors for.
     *  \param n_query_points The number of query points.
     *  \param qargs The query arguments that should be used to find neighbors.
     */
    std::shared_ptr<NeighborQueryIterator> query(const vec3<float>* query_points, unsigned int n_query_points,
                                                 QueryArgs query_args) const override
    {
        this->validateQueryArgs(query_args);

        // Use CellQuery for ball queries (can early terminate when shells exceed r_max)
        // Use AABBQuery for nearest queries (CellQuery must search all shells for correctness)
        if (query_args.mode == QueryType::ball)
        {
            if (!m_cell_query)
            {
                m_cell_query = std::make_unique<CellQuery>(m_box, m_points, m_n_points);
            }
            return m_cell_query->query(query_points, n_query_points, query_args);
        }
        else
        {
            if (!m_aabb_query)
            {
                m_aabb_query = std::make_unique<AABBQuery>(m_box, m_points, m_n_points);
            }
            return m_aabb_query->query(query_points, n_query_points, query_args);
        }
    }

    // dummy implementation for pure virtual function in the parent class
    std::shared_ptr<NeighborQueryPerPointIterator>
    querySingle(const vec3<float> query_point, unsigned int query_point_idx, QueryArgs qargs) const override
    {
        if (qargs.mode == QueryType::ball)
        {
            if (!m_cell_query)
            {
                throw std::runtime_error(
                    "The underlying CellQuery object has not yet been initialized. Please "
                    "report this error.");
            }
            return m_cell_query->querySingle(query_point, query_point_idx, qargs);
        }
        else
        {
            if (!m_aabb_query)
            {
                throw std::runtime_error(
                    "The underlying AABBQuery object has not yet been initialized. Please "
                    "report this error.");
            }
            return m_aabb_query->querySingle(query_point, query_point_idx, qargs);
        }
    }

private:
    mutable std::unique_ptr<CellQuery> m_cell_query; //!< Used for ball queries (efficient shell termination).
    mutable std::unique_ptr<AABBQuery>
        m_aabb_query; //!< Used for nearest queries (CellQuery must search all shells).
};

}; }; // end namespace freud::locality

#endif // NEIGHBOR_QUERY_H
