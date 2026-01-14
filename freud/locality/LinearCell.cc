// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "LinearCell.h"
#include <cmath>

namespace freud::locality {

//! Compute the number of cells along each cartesian direction, saving relevant data.
inline void CellQuery::setupGrid(const float r_cut)
{
    updateImageVectors(r_cut, true);
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
    const unsigned int total_cells = m_nx * m_ny * m_nz;
    // Allocate buffers
    m_counts.assign(total_cells, 0);      // Total occupancy of cell
    m_counts_real.assign(total_cells, 0); // Offsets for ghosts
    m_cell_starts.reserve(total_cells);   // Jumplist
    // NOTE: we do not know how many ghosts there are, so these are underestimates
    m_linear_buffer.reserve(m_n_total);
    std::vector<std::pair<int, TaggedPosition>> particle_cell_mapping(m_n_total);

    // Iterate over particles and images, adding ghosts where necessary
    for (size_t i = 0; i < m_n_total; i++)
    {
        const vec3<float> local_point = m_points[i];
        const vec3<float>* local_images = m_image_list.data();

        // Compute the axis-aligned (?) ellipse that represents our r_cut in fractional
        // coordinates.
        const vec3<float> plane_distances = m_box.getNearestPlaneDistance();
        const vec3<float> fractional_rcut = vec3<float>(r_cut, r_cut, r_cut) / plane_distances;

        for (size_t image_index = 0; image_index < m_n_images; image_index++)
        {
            const vec3<float> trial = local_point + local_images[image_index];
            const GhostPacket new_ghosts = generateGhosts(trial, fractional_rcut);
        }
    }
}
} // namespace freud::locality
