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
        : CellIterator(neighbor_query, query_point, query_point_idx, r_max, r_min, exclude_ii)
    {
        float m_r_max_squared = r_max * r_max;
        float m_r_min_squared = r_min * r_min;
        vec3<int> xyz = m_cell_query->cell_idx_xyz(query_point);
        unsigned int idx = m_cell_query->get_cell_idx(query_point);
        // TODO: this is only the current cell!
        m_cell_start_index = m_cell_query->m_cell_starts[idx];
        m_cell_end_offset = m_cell_query->m_counts[idx];
        m_cell_data = m_cell_query->m_linear_buffer.data() + m_cell_start_index;
        //     std::cout << "start index: " << m_cell_start_index << "\n";
        //     std::cout << "end offset:  " << m_cell_end_offset << "\n";
        //     std::cout << "m_cell_data:  " << m_cell_data << "\n";
    }

    //! Empty Destructor
    ~CellQueryBallIterator() override = default;

    //! Get the next element.
    NeighborBond next() override final;

protected:
    float m_r_max_sq;
    float m_r_min_sq;
    unsigned int m_query_point_idx;
    unsigned int m_idx = 0;
    unsigned int m_cell_start_index;
    unsigned int m_cell_end_offset;
    TaggedPosition* m_cell_data;
};
inline NeighborBond CellQueryBallIterator::next()
{
    if (m_idx >= m_cell_end_offset)
    {
        m_finished = true;
        return ITERATOR_TERMINATOR;
    }
    // TODO: handle exclude_ii
    TaggedPosition possible_neighbor = m_cell_data[m_idx++];
    const vec3<float> r_ij = possible_neighbor.p - m_query_point;
    float r_sq = dot(r_ij, r_ij);
    if (r_sq < m_r_max_sq && r_sq >= m_r_min_sq)
    {
        return NeighborBond(m_query_point_idx, possible_neighbor.particle_index, std::sqrt(r_sq), 1, r_ij);
    }
    // TODO: Implement neighbor search
}

} // namespace freud::locality
