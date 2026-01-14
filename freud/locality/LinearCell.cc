// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "LinearCell.h"
#include <cmath>

namespace freud::locality {

//! Cell list data unit.
struct TaggedParticle
{
    vec3<float> p;      //!< Position of the particle
    int particle_index; //!< Index of the particle (out of m_n_points, negative=ghost)
};

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
inline void CellQuery::buildGrid(const float r_cut)
{
    const unsigned int total_cells = m_nx * m_ny * m_nz;
    // Allocate buffers
    m_counts.assign(total_cells, 0);      // Total occupancy of cell
    m_counts_real.assign(total_cells, 0); // Offsets for ghosts
    m_cell_starts.reserve(total_cells);      // Jumplist
}
} // namespace freud::locality
