// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#pragma once

#include "NeighborQuery.h"
#include <vector>
/*! \file LinearCell.h
 *  \brief Build an cell list from points and query it for neighbors.
 */
namespace freud::locality {

// Forward declaration of iterator types we return from the query
class CellQueryBallIterator;

class CellQuery : public NeighborQuery
{
public:
    //! Constructs the compute
    CellQuery();

    //! New-style constructor. It's safe to inherit and use the parent class.
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
            validateNearestNeighborArgs(args);
        }
    }

    //! Compute the cell index of a point p, returning False for those outside the grid.
    bool get_cell_idx_safe(const vec3<float>& p, unsigned int& idx) const
    {
        int cx = static_cast<int>((p.x - m_min_pos.x) * m_cell_inverse_length);
        int cy = static_cast<int>((p.y - m_min_pos.y) * m_cell_inverse_length);
        int cz = static_cast<int>((p.z - m_min_pos.z) * m_cell_inverse_length);

        if (cx < 0 || cy < 0 || cz < 0 || cx >= m_nx || cy >= m_ny || cz >= m_nz)
        {
            return false;
        }

        idx = (cz * m_ny + cy) * m_nx + cx;
        return true;
    }
    void updateImageVectors(float r_max, bool _check_r_max)
    {
        m_image_list = freud::locality::updateImageVectors(m_box, r_max, _check_r_max, m_n_images);
    }

    float m_cell_inverse_length; //!< Reciprocal of r_cut, the width of each cell
    vec3<float> m_min_pos;       //!< Lower leftmost corner of the grid: box.m_lo - rcut

private:
    //! Cell list data unit.
    struct TaggedPosition
    {
        vec3<float> p;      //!< Position of the particle
        int particle_index; //!< Index of the particle (out of m_n_points, negative=ghost)
    };
    struct GhostPacket
    {
        std::array<vec3<int>, 7> images;
        unsigned int n_images = 0;
    };
    // std::array<TaggedPosition, 8> is_in_ghost_layer(const vec3<float>& point,
    //                                                 const vec3<float>& fractional_r_cut)
    // {
    //     vec3<float> f = m_box.makeFractional(point);

    //     // Check whether our point is within [-f_r_cut, 1+f_r_cut]
    //     bool in_x = (f.x >= -fractional_r_cut.x) && (f.x <= 1.0f + fractional_r_cut.x);
    //     bool in_y = (f.y >= -fractional_r_cut.y) && (f.y <= 1.0f + fractional_r_cut.y);
    //     bool in_z = (f.z >= -fractional_r_cut.z) && (f.z <= 1.0f + fractional_r_cut.z);

    //     // return in_x && in_y && in_z;
    // }
    GhostPacket generateGhosts(const vec3<float>& point, const vec3<float>& fractional_r_cut)
    {
        GhostPacket result;
        vec3<float> f = m_box.makeFractional(point);
        // Determine the single shift direction for each dimension (-1, 0, or 1)
        // We use integer arithmetic to avoid branching where possible, or simple ternaries.
        int dx = 0;
        if (f.x < fractional_r_cut.x)
        {
            dx = 1;
        }
        else if (f.x > 1.0f - fractional_r_cut.x)
        {
            dx = -1;
        }

        int dy = 0;
        if (f.y < fractional_r_cut.y)
        {
            dy = 1;
        }
        else if (f.y > 1.0f - fractional_r_cut.y)
        {
            dy = -1;
        }

        int dz = 0;
        if (f.z < fractional_r_cut.z)
        {
            dz = 1;
        }
        else if (f.z > 1.0f - fractional_r_cut.z)
        {
            dz = -1;
        }

        // Early exit if we are fully inside the bulk
        if (dx == 0 && dy == 0 && dz == 0)
        {
            return result;
        }

        // Explicitly check the 7 combinations.
        // For 3D cuboids, we place ghosts near the 6 faces, 12 edges, and 8 vertices.
        // However, because r_cut < L/2, we can be near a maximum of 3 faces, 3 edges,
        // and one vertex at a time for a max of 7 new ghosts per particle

        // Face Neighbors (one displacement)
        if (dx != 0)
        {
            result.images[result.n_images++] = {dx, 0, 0};
        }
        if (dy != 0)
        {
            result.images[result.n_images++] = {0, dy, 0};
        }
        if (dz != 0)
        {
            result.images[result.n_images++] = {0, 0, dz};
        }

        // Edge Neighbors (two displacements)
        if (dx != 0 && dy != 0)
        {
            result.images[result.n_images++] = {dx, dy, 0};
        }
        if (dx != 0 && dz != 0)
        {
            result.images[result.n_images++] = {dx, 0, dz};
        }
        if (dy != 0 && dz != 0)
        {
            result.images[result.n_images++] = {0, dy, dz};
        }

        // Corner Neighbor (all displacements populated)
        if (dx != 0 && dy != 0 && dz != 0)
        {
            result.images[result.n_images++] = {dx, dy, dz};
        }

        return result;
    }

    //! Compute the grid cell parameters.
    inline void setupGrid(const float r_cut);
    inline void buildGrid(const float r_cut);
    unsigned int m_nx;                           //!< Number of cells in the x dimension
    unsigned int m_ny;                           //!< Number of cells in the y dimension
    unsigned int m_nz;                           //!< Number of cells in the z dimension
    unsigned int m_n_total;                      //!< Total number of particles, including ghosts
    unsigned int m_n_images;                     //!< Total number of periodic images
    std::vector<unsigned int> m_counts;          //!< Number of particles in each cell
    std::vector<unsigned int> m_counts_real;     //!< Number of real particles in each cell
    std::vector<unsigned int> m_cell_starts;     //!< Position of each cell in the buffer
    std::vector<vec3<float>> m_image_list;       //!< Displacement vector to each image
    std::vector<TaggedPosition> m_linear_buffer; //!< Linear array of particles & ghosts

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
