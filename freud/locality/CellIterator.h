// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "LinearCell.h"
#include "NeighborQuery.h"
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
          m_cell_query(neighbor_query)
    {}

    //! Empty Destructor
    ~CellIterator() override = default;

    void updateImageVectors(float r_max, bool _check_r_max = true)
    {
        box::Box const box = m_neighbor_query->getBox();
        m_image_list = freud::locality::updateImageVectors(box, r_max, _check_r_max, m_n_images);
    }

protected:
    const CellQuery* m_cell_query;         //!< Link to the CellQuery object
    std::vector<vec3<float>> m_image_list; //!< List of translation vectors
    unsigned int m_n_images {0};           //!< The number of image vectors to check
};

//! Iterator that gets neighbors in a ball of size r_max using Cell tree structures.
class CellQueryBallIterator : public CellIterator
{
public:
    //! Constructor
    CellQueryBallIterator(const CellQuery* neighbor_query, const vec3<float>& query_point,
                          unsigned int query_point_idx, float r_max, float r_min, bool exclude_ii,
                          bool _check_r_max = true)
        : CellIterator(neighbor_query, query_point, query_point_idx, r_max, r_min, exclude_ii)
    {
        updateImageVectors(m_r_max, _check_r_max);
    }

    //! Empty Destructor
    ~CellQueryBallIterator() override = default;

    //! Get the next element.
    NeighborBond next() override;

private:
    unsigned int cur_image {0};    //!< The current node in the tree.
    unsigned int cur_node_idx {0}; //!< The current node in the tree.
    unsigned int cur_ref_p {
        0}; //!< The current index into the reference particles in the current node of the tree.
};
} // namespace freud::locality
