// Copyright (c) 2010-2026 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#pragma once

#include "NeighborQuery.h"
#include "VectorMath.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
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
class CellQueryNearestIterator;

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
        // Note we must floor the values so that small negatives correctly map to 0
        return {static_cast<int>(std::floor((p.x - m_min_pos.x) * m_cell_inverse_length)),
                static_cast<int>(std::floor((p.y - m_min_pos.y) * m_cell_inverse_length)),
                static_cast<int>(std::floor((p.z - m_min_pos.z) * m_cell_inverse_length))};
    }
    //! Compute the cell index of a point p, returning False for those outside the grid.
    bool getCellIdxSafe(const vec3<float>& p, unsigned int& idx) const
    {
        vec3<int> const xyz = cell_idx_xyz(p);
        int const cx = xyz.x;
        int const cy = xyz.y;
        int const cz = xyz.z;
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
        vec3<int> const xyz = cell_idx_xyz(p);
        return ((xyz.z * m_ny + xyz.y) * m_nx) + xyz.x;
    }

    //! Get the cell width (1/m_cell_inverse_length)
    float getCellWidth() const
    {
        return 1.0F / m_cell_inverse_length;
    }
    //! Get the cell width (1/m_cell_inverse_length)
    float getSafeRMax() const
    {
        return m_grid_max_safe_r_cut;
    }

    //! Get the number of real particles in each cell
    const std::vector<unsigned int>& getCountsReal() const
    {
        return m_counts_real;
    }

    //! Get the number of real+ghost particles in each cell
    const std::vector<unsigned int>& getCounts() const
    {
        return m_counts;
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

    //! Get the total number of particles (real + ghosts)
    unsigned int getNTotal() const
    {
        return m_n_total;
    }

    //! Get the cell starts array
    const std::vector<unsigned int>& getCellStarts() const
    {
        return m_cell_starts;
    }

    //! Compute the number of cells along each cartesian direction, saving relevant data.
    void setupGrid(const float r_cut) const
    {
        m_cell_inverse_length = 1.0F / r_cut;

        // Compute the widths of the box along each cartesian direction.
        float const w_x = m_box.getLx() + (m_box.getLy() * std::abs(m_box.getTiltFactorXY()))
            + (m_box.getLz() * std::abs(m_box.getTiltFactorXZ()));
        float const w_y = m_box.getLy() + (m_box.getLz() * std::abs(m_box.getTiltFactorYZ()));
        float const w_z = m_box.getLz();

        m_nx = static_cast<int>((w_x * m_cell_inverse_length)) + 3;
        m_ny = static_cast<int>((w_y * m_cell_inverse_length)) + 3;
        m_nz = static_cast<int>((w_z * m_cell_inverse_length)) + 3;

        // Compute the lowest, leftmost point on the grid
        float const box_min_x = (m_box.getLy() * std::min(0.0F, m_box.getTiltFactorXY()))
            + (m_box.getLz() * std::min(0.0F, m_box.getTiltFactorXZ()));

        float const box_min_y = m_box.getLz() * std::min(0.0F, m_box.getTiltFactorYZ());

        // Apply the ghost layer padding and shift to center the origin
        m_min_pos = {box_min_x - r_cut, box_min_y - r_cut, -r_cut};
        m_min_pos += m_box.makeAbsolute({0.0, 0.0, 0.0});
    }
    void buildGrid(const float r_cut) const;

    mutable std::vector<unsigned int> m_counts;          //!< Number of particles in each cell
    mutable std::vector<unsigned int> m_counts_real;     //!< Number of real particles in each cell
    mutable std::vector<unsigned int> m_cell_starts;     //!< Position of each cell in the buffer
    mutable std::vector<TaggedPosition> m_linear_buffer; //!< Linear array of particles & ghosts

protected:
    //! Validate the combination of specified arguments.
    void validateQueryArgs(QueryArgs& args) const override
    {
        NeighborQuery::validateQueryArgs(args);
        if (args.mode == QueryType::nearest)
        {
            validateNearestNeighborArgs(args);
        }

        // For nearest mode with infinite r_max, skip the box size validation: the grid
        // will be built with r_guess, which is checked for correctness in this->q uery
        if (args.mode != QueryType::nearest || !std::isinf(args.r_max))
        {
            // Validate r_max vs box size
            const vec3<float> nearest_plane_distance = m_box.getNearestPlaneDistance();
            if ((nearest_plane_distance.x <= args.r_max * 2.0F)
                || (nearest_plane_distance.y <= args.r_max * 2.0F)
                || (!m_box.is2D() && nearest_plane_distance.z <= args.r_max * 2.0F))
            {
                throw std::runtime_error("The CellQuery r_max is too large for this box.");
            }

            if (args.r_max <= 0)
            {
                throw std::invalid_argument("r_max must be positive.");
            }
        }

        if (!std::isinf(args.r_max) && args.r_max <= args.r_min)
        {
            throw std::invalid_argument("r_max must be greater than r_min.");
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
    GhostPacket generateGhosts(const vec3<float>& point, const vec3<float>& fractional_r_cut,
                               const vec3<float> Lx, const vec3<float> Ly, const vec3<float> Lz) const
    {
        GhostPacket result;
        const vec3<float> f = m_box.makeFractional(point);

        // Determine which images ∈ {-1, 0, 1} we are close enough to generate a ghost for
        // What we really want is a value that is -1 if near the high boundary, 1 if
        // near the low boundary, and 0 in the center of the box. This can be written as
        // ifs or ternaries, but it gets compiled to arithmetic most of the time anyway.

        int const dx = static_cast<int>(f.x < fractional_r_cut.x) - static_cast<int>(f.x > 1.0 - fractional_r_cut.x);
        int const dy = static_cast<int>(f.y < fractional_r_cut.y) - static_cast<int>(f.y > 1.0 - fractional_r_cut.y);
        int dz = static_cast<int>(f.z < fractional_r_cut.z) - static_cast<int>(f.z > 1.0 - fractional_r_cut.z);
        // Cannot have ghosts in a non-existent dimension
        if (m_box.is2D())
        {
            dz = 0;
        }
        // For particle in the bulk, we don't need to try and generate ghosts.
        if (dx == 0 && dy == 0 && dz == 0)
        {
            return result;
        }

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

    mutable bool m_built = false;              //!< Whether the grid has been built.
    mutable float m_grid_r_cut = 0.0;          //!< Current grid width to check if rebuild is necessary
    mutable float m_grid_max_safe_r_cut = 0.0; //!< Maximum safe r_cut for the grid

    //! Place particles into sorted linear buffer using cell starts and counts.
    void placeParticlesInSortedOrder(
        const std::vector<std::pair<unsigned int, TaggedPosition>>& particle_cell_mapping,
                                     std::vector<TaggedPosition>& sorted) const;

    friend class CellQueryBallIterator;
};

} // namespace freud::locality
