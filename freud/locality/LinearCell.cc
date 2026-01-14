// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "LinearCell.h"
#include <cmath>

namespace freud::locality {

//! Compute the number of cells along each cartesian direction, saving relevant data.
inline void CellQuery::setupGrid(const float r_cut)
{
    m_cell_inverse_length = 1.0f / r_cut;

    // Compute the boundary of our cell list
    // TODO: handle points outside box?
    vec3<float> pad = {r_cut, r_cut, r_cut};
    vec3<float> max_pos = m_box.getMaxCoord() + pad;
    m_min_pos = m_box.getMinCoord() - pad;

    // Number of cells in each dimension is ⌈(max-min) / r_cut⌉
    vec3<float> diagonal = (max_pos - m_min_pos);
    m_nx = static_cast<unsigned int>(std::ceil(diagonal.x * m_cell_inverse_length));
    m_ny = static_cast<unsigned int>(std::ceil(diagonal.y * m_cell_inverse_length));
    m_nz = static_cast<unsigned int>(std::ceil(diagonal.z * m_cell_inverse_length));
}
/*! Build and populate the cell list grid.
 *
 * AABBQuery implements its image logic in AABBIterator::next, which loops through
 * ghosts to find neighbors lazily. However, it is advantageous to compute ghosts once
 * and assign them to cells: this should make it possible for next() to literally just
 * increment a pointer which is optimal.
 */
inline void CellQuery::buildGrid(const float r_cut)
{
    const unsigned int total_n_cells = m_nx * m_ny * m_nz;
    // Allocate buffers
    m_counts.assign(total_n_cells, 0);      // Total occupancy of cell
    m_counts_real.assign(total_n_cells, 0); // Offsets for ghosts
    m_cell_starts.reserve(total_n_cells);   // Jumplist
    // NOTE: we do not know how many ghosts there are, so these are underestimates
    m_linear_buffer.reserve(m_n_points);
    // Cell index and TaggedPosition, which itself contains a particle/ghost index
    std::vector<std::pair<int, TaggedPosition>> particle_cell_mapping(m_n_points);

    // Iterate over particles and images, adding ghosts where necessary
    for (size_t i = 0; i < m_n_points; i++)
    {
        const vec3<float> local_point = m_points[i];

        // Compute the axis-aligned (?) ellipse that represents our r_cut in fractional
        // coordinates.
        const vec3<float> plane_distances = m_box.getNearestPlaneDistance();
        const vec3<float> fractional_rcut = vec3<float>(r_cut, r_cut, r_cut) / plane_distances;

        // TODO: SoA?
        const GhostPacket ghosts = generateGhosts(local_point, fractional_rcut);

        // NOTE: this will fail if i is > INT_MAX ( 4 billion )
        const int ghost_tag = -static_cast<int>(i);
        for (size_t ghost_index = 0; ghost_index < ghosts.n_displacements; ghost_index++)
        {
            const vec3<float> ghost = local_point + ghosts.displacements[ghost_index];
            unsigned int idx;
            if (get_cell_idx_safe(ghost, idx))
            {
                TaggedPosition p = {ghost, ghost_tag};
                particle_cell_mapping.push_back({idx, p});
                m_n_total++;
            }
        }
        // TODO: is query point always within cell?
        unsigned int idx;
        get_cell_idx_safe(local_point, idx);
        TaggedPosition p = {local_point, static_cast<int>(i)};
        particle_cell_mapping.push_back({idx, p});
        m_n_total++;
    }
    // Calculate starts array (prefix sum)
    int off = 0;
    for (int c = 0; c < total_n_cells; ++c)
    {
        m_starts[c] = off;
        off += m_counts[c];
    }
}
} // namespace freud::locality
