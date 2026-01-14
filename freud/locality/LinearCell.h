#pragma once

#include "NeighborQuery.h"
#include <vector>
/*! \file LinearCell.h
 *  \brief Build an cell list from points and query it for neighbors.
 */
namespace freud::locality {

// forward declaration of iterator types we return from the query
class CellQueryBallIterator;

class CellQuery : public NeighborQuery
{
public:
    //! Constructs the compute
    CellQuery();

    //! New-style constructor.
    CellQuery(const box::Box& box, const vec3<float>* points, unsigned int n_points);

    //! Destructor
    ~CellQuery() override;

    //! Implementation of per-particle query for CellQuery (see NeighborQuery.h for documentation).
    /*! \param query_point The point to find neighbors for.
     *  \param n_query_points The number of query points.
     *  \param qargs The query arguments that should be used to find neighbors.
     */
    std::shared_ptr<NeighborQueryPerPointIterator>
    querySingle(const vec3<float> query_point, unsigned int query_point_idx, QueryArgs args) const final;

    // TODO: taggedparticle?
    std::vector<int> m_particles; //!< Linear map of particle indices in memory

protected:
    //! Validate the combination of specified arguments.
    /*! Add to parent function to account for the various arguments
     *  specifically required for CellQuery nearest neighbor queries.
     */
    void validateQueryArgs(QueryArgs& args) const override
    {
        NeighborQuery::validateQueryArgs(args);
        if (args.mode == QueryType::nearest)
        {
            if (args.scale == DEFAULT_SCALE)
            {
                args.scale = float(1.1);
            }
            else if (args.scale <= float(1.0))
            {
                throw std::runtime_error("The scale query argument must be greater than 1.");
            }

            if (args.r_guess == DEFAULT_R_GUESS)
            {
                // By default, we assume a homogeneous system density and use
                // that to estimate the distance we need to query. This
                // calculation assumes a constant density of N/V, where N is
                // the number of particles and V is the box volume, and it
                // calculates the radius of a sphere that will contain the
                // desired number of neighbors.

                float const r_guess = std::cbrtf(
                    (float(3.0) * static_cast<float>(args.num_neighbors) * m_box.getVolume())
                    / (float(4.0) * static_cast<float>(M_PI) * static_cast<float>(getNPoints())));

                // The upper bound is set by the minimum nearest plane distances.
                vec3<float> const nearest_plane_distance = m_box.getNearestPlaneDistance();
                float min_plane_distance = std::min(nearest_plane_distance.x, nearest_plane_distance.y);
                if (!m_box.is2D())
                {
                    min_plane_distance = std::min(min_plane_distance, nearest_plane_distance.z);
                }

                args.r_guess = std::min(r_guess, min_plane_distance / float(2.0));
            }
            args.r_guess = std::min(args.r_guess, args.r_max);
        }
    }

private:
    // ! Driver for tree configuration
    // void setupTree(unsigned int N);

    //! Maps particles by local id to their id within their type trees
    // void mapParticlesByType();

    //! Driver to build Cell trees
    // void buildTree(const vec3<float>* points, unsigned int N);

    // std::vector<Cell> m_aabbs; //!< Flat array of Cells of all types
};

} // namespace freud::locality

// Include CellIterator.h after CellQuery is fully defined to avoid circular dependency.
// This provides the complete definition of CellQueryBallIterator needed by querySingle.
#include "CellIterator.h"

namespace freud::locality {

// Implementation of querySingle - must be after including CellIterator.h
// so that CellQueryBallIterator is fully defined
inline std::shared_ptr<NeighborQueryPerPointIterator>
CellQuery::querySingle(const vec3<float> query_point, unsigned int query_point_idx, QueryArgs args) const
{
    this->validateQueryArgs(args);
    if (args.mode == QueryType::ball)
    {
        // TODO
        return std::make_shared<CellQueryBallIterator>(this, query_point, query_point_idx, args.r_max,
                                                       args.r_min, args.exclude_ii);
    }
    // if (args.mode == QueryType::nearest)
    // {
    //     // TODO
    //     return std::make_shared<CellQueryIterator>(this, query_point, query_point_idx,
    //     args.num_neighbors,
    //                                                args.r_guess, args.r_max, args.r_min, args.scale,
    //                                                args.exclude_ii);
    // }
    throw std::runtime_error("Invalid query mode provided to query function in CellQuery.");
}

} // namespace freud::locality
