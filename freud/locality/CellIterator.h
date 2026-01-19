// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#pragma once
#include "LinearCell.h"
#include "NeighborBond.h"
#include "NeighborQuery.h"
#include "VectorMath.h"
#include <iostream>
#include <map>
#include <stdexcept>

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
        : CellIterator(neighbor_query, query_point, query_point_idx, r_max, r_min, exclude_ii)
    {
        if (m_cell_query->m_linear_buffer.data() == nullptr)
        {
            throw std::runtime_error("Cell data is uninitialized.");
        }

        m_r_max_sq = r_max * r_max;
        m_r_min_sq = r_min * r_min;

        // Check if the query point is within the grid bounds. TODO: is this allowed?
        unsigned int cell_idx_u;
        if (!m_cell_query->getCellIdxSafe(m_query_point, cell_idx_u))
        {
            m_finished = true;
            return;
        }

        vec3<int> coords = m_cell_query->cell_idx_xyz(m_query_point);
        m_cx = coords.x;
        m_cy = coords.y;
        m_cz = coords.z;

        // Initialize state for the loop
        m_dy = -1;
        m_dz = -1;
        m_particle_idx = -1;
        m_list_end = -1;

        find_next_pair();
    }

    //! Empty Destructor
    ~CellQueryBallIterator() override = default;

    //! Get the next element.
    NeighborBond next() override final
    {
        NeighborBond res = m_current_bond;
        find_next_pair();
        return res;
    }

private:
    void find_next_pair()
    {
        int particle_idx = m_particle_idx;
        int start_dz = m_dz;
        int start_dy = m_dy;

        // Cache grid dimensions and pointers for performance
        const int nz_dim = static_cast<int>(m_cell_query->getNz());
        const int ny_dim = static_cast<int>(m_cell_query->getNy());
        const int nx_dim = static_cast<int>(m_cell_query->getNx());

        const auto* starts_data = m_cell_query->m_cell_starts.data();
        const auto* counts_data = m_cell_query->m_counts.data();

        for (int dz = start_dz; dz <= 1; dz++)
        {
            int nz = m_cz + dz;
            if (nz >= 0 && nz < nz_dim)
            {
                for (int dy = start_dy; dy <= 1; dy++)
                {
                    int ny = m_cy + dy;
                    if (ny >= 0 && ny < ny_dim)
                    {
                        // If starting a new row, determine the contiguous particle range
                        if (particle_idx == -1)
                        {
                            // Compute base index for the row (dx=0)
                            // LinearCell idx = (cz * Ny + cy) * Nx + cx
                            int row_cell_idx = ((nz * ny_dim) + ny) * nx_dim;
                            // Adjust for center column
                            row_cell_idx += m_cx;

                            // Determine valid dx range [-1, 1] clipped to [0, Nx-1] relative to global
                            // Since row_cell_idx is at 'cx', we check bounds of cx+dx
                            int min_dx = (m_cx > 0) ? -1 : 0;
                            int max_dx = (m_cx < nx_dim - 1) ? 1 : 0;

                            // Calculate start and end indices of the contiguous block
                            int start_cell_idx = row_cell_idx + min_dx;
                            int end_cell_idx = row_cell_idx + max_dx;

                            particle_idx = starts_data[start_cell_idx];
                            m_list_end = starts_data[end_cell_idx] + counts_data[end_cell_idx];
                        }

                        for (; particle_idx < m_list_end; ++particle_idx)
                        {
                            if (test_particle(particle_idx, dy, dz))
                            {
                                return;
                            }
                        }
                        particle_idx = -1;
                    }
                }
            }
            start_dy = -1;
        }
        m_finished = true;
        m_current_bond = ITERATOR_TERMINATOR;
    }

    bool test_particle(int particle_idx, int dy, int dz)
    {
        const TaggedPosition& p = m_cell_query->m_linear_buffer[particle_idx];

        const int neighbor_idx_raw = p.particle_index;
        const unsigned int real_id = (neighbor_idx_raw ^ (neighbor_idx_raw >> 31));

        if (this->m_exclude_ii && real_id == this->m_query_point_idx)
        {
            return false;
        }

        vec3<float> delta = p.p - this->m_query_point;
        float d2 = dot(delta, delta);

        if (d2 < m_r_max_sq && d2 >= m_r_min_sq)
        {
            m_current_bond = NeighborBond(this->m_query_point_idx, real_id, std::sqrt(d2), 1.0f, delta);
            m_dy = dy;
            m_dz = dz;
            m_particle_idx = particle_idx + 1;
            return true;
        }
        return false;
    }

    float m_r_max_sq;
    float m_r_min_sq;

    // Iterator state
    int m_cx, m_cy, m_cz; // Cell coordinates of query point
    int m_dy, m_dz;       // Current cell offset being searched
    int m_particle_idx;   // Current particle index in linear buffer
    int m_list_end;       // End index of the current contiguous block
    NeighborBond m_current_bond;
};

//! Iterator that gets the k-nearest neighbors using the CellQuery grid structure.
class CellQueryNearestIterator : public CellIterator
{
public:
    //! Constructor
    CellQueryNearestIterator(const CellQuery* neighbor_query, const vec3<float>& query_point,
                             unsigned int query_point_idx, unsigned int num_neighbors, float r_max,
                             float r_min, bool exclude_ii)
        : CellIterator(neighbor_query, query_point, query_point_idx, r_max, r_min, exclude_ii),
          m_k(num_neighbors), m_r_min_sq(r_min * r_min), m_r_max_sq(r_max * r_max)
    {
        if (m_cell_query->m_linear_buffer.data() == nullptr)
        {
            throw std::runtime_error("Cell data is uninitialized.");
        }

        // Check if the query point is within the grid bounds.
        unsigned int cell_idx_u;
        if (!m_cell_query->getCellIdxSafe(m_query_point, cell_idx_u))
        {
            m_finished = true;
            return;
        }

        const vec3<int> coords = m_cell_query->cell_idx_xyz(m_query_point);
        const int cx = coords.x;
        const int cy = coords.y;
        const int cz = coords.z;

        // Cache grid dimensions and pointers for performance
        const int nz_dim = static_cast<int>(m_cell_query->getNz());
        const int ny_dim = static_cast<int>(m_cell_query->getNy());
        const int nx_dim = static_cast<int>(m_cell_query->getNx());

        // Map to track the minimum distance for each particle ID
        std::map<unsigned int, NeighborBond> min_distance_bonds;

        // Collect all neighbors within r_max, keeping only the closest image of each particle
        for (int dz = -1; dz <= 1; dz++)
        {
            const int nz = cz + dz;
            if (nz >= 0 && nz < nz_dim)
            {
                for (int dy = -1; dy <= 1; dy++)
                {
                    const int ny = cy + dy;
                    if (ny >= 0 && ny < ny_dim)
                    {
                        // Determine valid dx range [-1, 1] clipped to [0, Nx-1] relative to global
                        const int min_dx = (cx > 0) ? -1 : 0;
                        const int max_dx = (cx < nx_dim - 1) ? 1 : 0;

                        // Calculate start and end indices of the contiguous block
                        const int start_cell_idx = (((nz * ny_dim) + ny) * nx_dim) + cx + min_dx;
                        const int end_cell_idx = (((nz * ny_dim) + ny) * nx_dim) + cx + max_dx;

                        // Process the contiguous block of cells
                        processCell(min_distance_bonds, start_cell_idx, end_cell_idx);
                    }
                }
            }
        }
        // If we don't have enough neighbors, search the second shell
        if (min_distance_bonds.size() < m_k)
        {
            for (int dz = -2; dz <= 2; dz++)
            {
                const int nz = cz + dz;
                if (nz < 0 || nz >= nz_dim)
                {
                    continue;
                }

                for (int dy = -2; dy <= 2; dy++)
                {
                    const int ny = cy + dy;
                    if (ny < 0 || ny >= ny_dim)
                    {
                        continue;
                    }

                    for (int dx = -2; dx <= 2; dx++)
                    {
                        // Only process cells in the second shell (surface of 5x5x5 cube)
                        if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != 2)
                        {
                            continue;
                        }

                        const int nx = cx + dx;
                        if (nx < 0 || nx >= nx_dim)
                        {
                            continue;
                        }

                        // Process a single cell at a time
                        const int cell_idx = (((nz * ny_dim) + ny) * nx_dim) + nx;
                        processCell(min_distance_bonds, cell_idx, cell_idx, true);
                    }
                }
            }
        }

        // Extract unique bonds from the map and sort by distance
        m_neighbors.reserve(min_distance_bonds.size());
        for (const auto& entry : min_distance_bonds)
        {
            m_neighbors.push_back(entry.second);
        }
        std::sort(m_neighbors.begin(), m_neighbors.end());
    }

    //! Empty Destructor
    ~CellQueryNearestIterator() override = default;

    //! Get the next element.
    NeighborBond next() override
    {
        if (m_cur_idx < m_neighbors.size() && m_cur_idx < m_k)
        {
            return m_neighbors[m_cur_idx++];
        }
        m_finished = true;
        return ITERATOR_TERMINATOR;
    }

private:
    //! Process a contiguous block of cells and add neighbors to the map.
    void processCell(std::map<unsigned int, NeighborBond>& min_distance_bonds, int start_cell_idx,
                     const int end_cell_idx, const bool wrap = false)
    {
        const auto* starts_data = m_cell_query->m_cell_starts.data();
        const auto* counts_data = m_cell_query->m_counts.data();
        const auto* buffer_data = m_cell_query->m_linear_buffer.data();

        // x-1...x+1 cells are contiguous in memory so we can take the entire segment
        int particle_idx = starts_data[start_cell_idx];
        const int list_end = starts_data[end_cell_idx] + counts_data[end_cell_idx];

        for (; particle_idx < list_end; ++particle_idx)
        {
            const TaggedPosition& p = buffer_data[particle_idx];

            const int neighbor_idx_raw = p.particle_index;
            const unsigned int real_id = (neighbor_idx_raw ^ (neighbor_idx_raw >> 31));

            if (this->m_exclude_ii && real_id == this->m_query_point_idx)
            {
                continue;
            }

            // Compute delta and apply periodic wrapping for minimum image convention
            vec3<float> delta = p.p - this->m_query_point;
            delta = wrap ? m_cell_query->getBox().wrap(delta) : delta;
            const float d2 = dot(delta, delta);

            if (d2 < m_r_max_sq && d2 >= m_r_min_sq)
            {
                const float distance = std::sqrt(d2);

                NeighborBond bond(this->m_query_point_idx, real_id, distance, 1.0f, delta);

                // Keep only the closest image of each particle
                auto it = min_distance_bonds.find(real_id);
                if (it == min_distance_bonds.end() || distance < it->second.getDistance())
                {
                    min_distance_bonds[real_id] = bond;
                }
            }
        }
    }

    std::vector<NeighborBond> m_neighbors;
    unsigned int m_cur_idx {0};
    unsigned int m_k;
    float m_r_max_sq;
    float m_r_min_sq;
};

} // namespace freud::locality
