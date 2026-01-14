// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "LinearCell.h"
#include <cmath>

namespace freud::locality {

//! Compute the number of cells along each cartesian direction, saving relevant data.
void CellQuery::makeGrid(const float r_cut)
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

} // namespace freud::locality
