// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#pragma once
#include "LinearCell.h"
#include "NeighborBond.h"
#include "NeighborQuery.h"
#include "VectorMath.h"
#include <iostream>
#include <stdexcept>
// Iterator structure
namespace freud::locality {

class CellIterator : public NeighborQueryPerPointIterator
{
public:
    //! Constructor
    CellIterator(const CellQuery* neighbor_query, const vec3<float>& query_point,
                 unsigned int query_point_idx, float r_max, float r_min, bool exclude_ii)
        : NeighborQueryPerPointIterator(neighbor_query, query_point, query_point_idx, r_max, r_min,
                                        exclude_ii),
          m_cell_query(neighbor_query), m_query_point_idx(query_point_idx)
    {}

    //! Empty Destructor
    ~CellIterator() override = default;

protected:
    const CellQuery* m_cell_query; //!< Link to the CellQuery object
    unsigned int m_query_point_idx;
};

//! Iterator that gets neighbors in a ball of size r_max using Cell tree structures.
class CellQueryBallIterator : public CellIterator
{
public:
    //! Constructor
    CellQueryBallIterator(const CellQuery* neighbor_query, const vec3<float>& query_point,
                          unsigned int query_point_idx, float r_max, float r_min, bool exclude_ii)
        : CellIterator(neighbor_query, query_point, query_point_idx, r_max, r_min, exclude_ii),
          m_r_max_sq(r_max * r_max), m_r_min_sq(r_min * r_min)
    {
        if (m_cell_query->m_linear_buffer.data() == nullptr)
        {
            throw std::runtime_error("Cell data is uninitialized.");
        }

        // Get current cell XYZ coordinates
        vec3<int> center_cell = m_cell_query->cell_idx_xyz(query_point);

        // Search the 3x3x3 neighboring cells. TODO: order
        for (int dz = -1; dz <= 1; ++dz)
        {
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dx = -1; dx <= 1; ++dx)
                {
                    vec3<int> neighbor_xyz = center_cell + vec3<int>(dx, dy, dz);

                    // Check bounds
                    if (neighbor_xyz.x >= 0 && neighbor_xyz.x < m_cell_query->getNx() && neighbor_xyz.y >= 0
                        && neighbor_xyz.y < m_cell_query->getNy() && neighbor_xyz.z >= 0
                        && neighbor_xyz.z < m_cell_query->getNz())
                    {
                        // Convert to flat index
                        unsigned int cell_idx = (neighbor_xyz.z * m_cell_query->getNy() + neighbor_xyz.y)
                                * m_cell_query->getNx()
                            + neighbor_xyz.x;

                        // Only add cells that have particles
                        if (m_cell_query->m_counts[cell_idx] > 0)
                        {
                            CellInfo cell_info;
                            cell_info.data = m_cell_query->m_linear_buffer.data()
                                + m_cell_query->m_cell_starts[cell_idx];
                            cell_info.count = m_cell_query->m_counts[cell_idx];
                            m_cells_to_search.push_back(cell_info);
                        }
                    }
                }
            }
        }
    }

    //! Empty Destructor
    ~CellQueryBallIterator() override = default;

    //! Get the next element.
    NeighborBond next() override final;

protected:
    struct CellInfo
    {
        TaggedPosition* data;
        unsigned int count;
    };

    float m_r_max_sq;
    float m_r_min_sq;
    unsigned int m_query_point_idx;

    std::vector<CellInfo> m_cells_to_search;
    size_t m_current_cell_idx = 0;
    size_t m_current_particle_idx = 0;
};
inline NeighborBond CellQueryBallIterator::next()
{
    while (m_current_cell_idx < m_cells_to_search.size())
    {
        const CellInfo& current_cell = m_cells_to_search[m_current_cell_idx];

        while (m_current_particle_idx < current_cell.count)
        {
            TaggedPosition possible_neighbor = current_cell.data[m_current_particle_idx++];

            // Handle exclude_ii - ghost particles have negative indices
            if (m_exclude_ii)
            {
                int neighbor_idx = possible_neighbor.particle_index;
                if (neighbor_idx == static_cast<int>(m_query_point_idx)
                    || neighbor_idx == ~static_cast<int>(m_query_point_idx))
                {
                    continue;
                }
            }

            const vec3<float> r_ij = possible_neighbor.p - m_query_point;
            float r_sq = dot(r_ij, r_ij);

            if (r_sq < m_r_max_sq && r_sq >= m_r_min_sq)
            {
                // For ghost particles (negative indices), return the original positive index
                unsigned int neighbor_idx = (possible_neighbor.particle_index < 0)
                    ? static_cast<unsigned int>(possible_neighbor.particle_index)
                    : static_cast<unsigned int>(possible_neighbor.particle_index);
                return NeighborBond(m_query_point_idx, neighbor_idx, std::sqrt(r_sq), 1, r_ij);
            }
        }

        // Move to next cell
        m_current_cell_idx++;
        m_current_particle_idx = 0;
    }

    m_finished = true;
    return ITERATOR_TERMINATOR;
}

} // namespace freud::locality
