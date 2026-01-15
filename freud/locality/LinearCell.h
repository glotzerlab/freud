// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#pragma once

#include "NeighborQuery.h"
#include <iostream>
#include <stdexcept>
#include <vector>
/*! \file LinearCell.h
 *  \brief Build an cell list from points and query it for neighbors.
 */
namespace freud::locality {
//! Cell list data unit.
struct TaggedPosition
{
    vec3<float> p;      //!< Position of the particle
    int particle_index; //!< Index of the particle (out of m_n_points, negative=ghost)
};

// Forward declaration of iterator types we return from the query
class CellQueryBallIterator;

class CellQuery : public NeighborQuery
{
public:
    //! Inherit constructors from NeighborQuery
    using NeighborQuery::NeighborQuery;

    //! Default constructor
    CellQuery() = default;

    //! Destructor
    ~CellQuery() override = default;

    //! Implementation of per-particle query for CellQuery (see NeighborQuery.h for documentation).
    /*! \param query_point The point to find neighbors for.
     *  \param n_query_points The number of query points.
     *  \param qargs The query arguments that should be used to find neighbors.
     */
    std::shared_ptr<NeighborQueryPerPointIterator>
    querySingle(const vec3<float> query_point, unsigned int query_point_idx, QueryArgs args) const final;

    //! Perform a query based on a set of query parameters.
    std::shared_ptr<NeighborQueryIterator> query(const vec3<float>* query_points, unsigned int n_query_points,
                                                 QueryArgs query_args) const override;

    vec3<int> cell_idx_xyz(const vec3<float>& p) const
    {
        return {static_cast<int>((p.x - m_min_pos.x) * m_cell_inverse_length),
                static_cast<int>((p.y - m_min_pos.y) * m_cell_inverse_length),
                static_cast<int>((p.z - m_min_pos.z) * m_cell_inverse_length)};
    }
    //! Compute the cell index of a point p, returning False for those outside the grid.
    bool getCellIdxSafe(const vec3<float>& p, unsigned int& idx) const
    {
        vec3<int> xyz = cell_idx_xyz(p);
        int cx = xyz.x;
        int cy = xyz.y;
        int cz = xyz.z;
        if (cx < 0 || cy < 0 || cz < 0 || cx >= m_nx || cy >= m_ny || cz >= m_nz)
        {
            return false;
        }

        idx = (cz * m_ny + cy) * m_nx + cx;
        return true;
    }
    //! Compute the cell index of a point p, returning False for those outside the grid.
    unsigned int getCellIdx(const vec3<float>& p) const
    {
        vec3<int> xyz = cell_idx_xyz(p);
        return ((xyz.z * m_ny + xyz.y) * m_nx) + xyz.x;
    }

    //! Get the cell width (1/m_cell_inverse_length)
    float getCellWidth() const
    {
        return 1.0f / m_cell_inverse_length;
    }

    //! Get the number of real particles in each cell
    const std::vector<unsigned int>& getRealCounts() const
    {
        return m_counts_real;
    }

    //! Get the lower leftmost corner of the grid
    std::vector<float> getMinPos() const
    {
        return {m_min_pos.x, m_min_pos.y, m_min_pos.z};
    }

    //! Get the inverse cell width
    float getCellInverseWidth() const
    {
        return m_cell_inverse_length;
    }

    //! Get the number of cells in the x dimension
    unsigned int getNx() const
    {
        return m_nx;
    }

    //! Get the number of cells in the y dimension
    unsigned int getNy() const
    {
        return m_ny;
    }

    //! Get the number of cells in the z dimension
    unsigned int getNz() const
    {
        return m_nz;
    }
    //! Compute the number of cells along each cartesian direction, saving relevant data.
    void setupGrid(const float r_cut) const
    {
        m_cell_inverse_length = 1.0f / r_cut;
        // Compute the widths of the box along each cartesian direction.
        float w_x = m_box.getLx() + (m_box.getLy() * m_box.getTiltFactorXY())
            + (m_box.getLz() * m_box.getTiltFactorXZ());
        float w_y = m_box.getLy() + (m_box.getLz() * m_box.getTiltFactorYZ());
        float w_z = m_box.getLz();

        m_nx = static_cast<int>((w_x * m_cell_inverse_length)) + 3;
        m_ny = static_cast<int>((w_y * m_cell_inverse_length)) + 3;
        m_nz = static_cast<int>((w_z * m_cell_inverse_length)) + 3;

        // Lowest, leftmost point on the grid
        m_min_pos.x = -0.5f * static_cast<float>(m_nx) * r_cut;
        m_min_pos.y = -0.5f * static_cast<float>(m_ny) * r_cut;
        m_min_pos.z = -0.5f * static_cast<float>(m_nz) * r_cut;
    }
    inline void buildGrid(const float r_cut) const;

    mutable std::vector<unsigned int> m_counts;          //!< Number of particles in each cell
    mutable std::vector<unsigned int> m_counts_real;     //!< Number of real particles in each cell
    mutable std::vector<unsigned int> m_cell_starts;     //!< Position of each cell in the buffer
    mutable std::vector<TaggedPosition> m_linear_buffer; //!< Linear array of particles & ghosts

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

    mutable float m_cell_inverse_length; //!< Reciprocal of r_cut, the width of each cell
    mutable vec3<float> m_min_pos;       //!< Lower leftmost corner of the grid: box.m_lo - rcut

private:
    //! Buffer of ghost particles for a single real particle.
    struct GhostPacket
    {
        std::array<vec3<float>, 7> displacements;
        unsigned int n_displacements = 0;
    };
    //! Compute the vectors mapping a point in the box to a point in (up to) 27 images.
    // This is adapted from updateImageVectors, and aims to save memory bandwidth by
    // calculating lattice vectors extra times.
    GhostPacket generateGhosts(const vec3<float>& point, const vec3<float>& fractional_r_cut) const
    {
        GhostPacket result;
        const vec3<float> f = m_box.makeFractional(point);
        int dx, dy, dz;
        // Determine which images ∈ {-1, 0, 1} we are close enough to generate a ghost for
        if (f.x < fractional_r_cut.x)
        {
            dx = 1;
        }
        else
        {
            dx = (f.x > 1.0f - fractional_r_cut.x) ? -1 : 0;
        }

        if (f.y < fractional_r_cut.y)
        {
            dy = 1;
        }
        else
        {
            dy = (f.y > 1.0f - fractional_r_cut.y) ? -1 : 0;
        }

        if (f.z < fractional_r_cut.z)
        {
            dz = 1;
        }
        else
        {
            dz = (f.z > 1.0f - fractional_r_cut.z) ? -1 : 0;
        }
        // Cannot have ghosts in a non-existent dimension
        if (!m_box.is2D())
        {
            dz = 0;
        }

        // For particle in the bulk, we don't need to try and generate ghosts.
        if (dx == 0 && dy == 0 && dz == 0)
        {
            return result;
        }

        // Compute lattice vectors that displace our input to generate a ghost.
        const vec3<float> Lx = m_box.getLatticeVector(0);
        const vec3<float> Ly = m_box.getLatticeVector(1);
        // Get zeros if we are in 2d
        const vec3<float> Lz = (!m_box.is2D()) ? m_box.getLatticeVector(2) : vec3<float>(0, 0, 0);

        auto new_site = [&](int i, int j, int k) {
            vec3<float> shift(0, 0, 0);
            // If {i,j,k}=={1} we have a positive displacement, otherwise negative.
            if (i != 0)
            {
                shift += (i == 1) ? Lx : -Lx;
            }
            if (j != 0)
            {
                shift += (j == 1) ? Ly : -Ly;
            }
            if (k != 0)
            {
                shift += (k == 1) ? Lz : -Lz;
            }
            result.displacements[result.n_displacements++] = shift;
        };

        // Face Neighbors
        if (dx != 0)
        {
            new_site(dx, 0, 0);
        }
        if (dy != 0)
        {
            new_site(0, dy, 0);
        }
        if (dz != 0)
        {
            new_site(0, 0, dz);
        }

        // Edge Neighbors
        if (dx != 0 && dy != 0)
        {
            new_site(dx, dy, 0);
        }
        if (dx != 0 && dz != 0)
        {
            new_site(dx, 0, dz);
        }
        if (dy != 0 && dz != 0)
        {
            new_site(0, dy, dz);
        }

        // Corner Neighbor
        if (dx != 0 && dy != 0 && dz != 0)
        {
            new_site(dx, dy, dz);
        }

        return result;
    }

    mutable unsigned int m_nx = 0;       //!< Number of cells in the x dimension
    mutable unsigned int m_ny = 0;       //!< Number of cells in the y dimension
    mutable unsigned int m_nz = 0;       //!< Number of cells in the z dimension
    mutable unsigned int m_n_total = 0;  //!< Total number of particles, including ghosts
    mutable unsigned int m_n_images = 0; //!< Total number of periodic images

    //! Maps particles by local id to their id within their type trees
    // void mapParticlesByType();

    //! Driver to build Cell trees
    // void buildTree(const vec3<float>* points, unsigned int N);

    // std::vector<Cell> m_aabbs; //!< Flat array of Cells of all types

    friend class CellQueryBallIterator;
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
    throw std::runtime_error("in querySIngle");
    this->validateQueryArgs(args);
    if (args.mode == QueryType::ball)
    {
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
