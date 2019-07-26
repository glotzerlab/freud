// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef VORONOI_H
#define VORONOI_H

#include "Box.h"
#include "NeighborList.h"
#include "VectorMath.h"

namespace freud { namespace locality {

struct NeighborBond
{
    NeighborBond() : index_i(0), index_j(0), weight(0), distance(0) {}

    NeighborBond(unsigned int index_i, unsigned int index_j, float w, float d)
        : index_i(index_i), index_j(index_j), weight(w), distance(d)
    {}

    unsigned int index_i; //! The point id.
    unsigned int index_j; //! The reference point id.
    float weight;         //! The weight of this bond.
    float distance;       //! The distance bewteen the points.
};

class Voronoi
{
public:
    // default constructor
    Voronoi();

    void compute(const box::Box& box, const vec3<double>* vertices, const int* ridge_points,
                 const int* ridge_vertices, unsigned int n_ridges, unsigned int N, const int* expanded_ids,
                 const vec3<double>* expanded_points, const int* ridge_vertex_indices);

    NeighborList* getNeighborList()
    {
        return &m_neighbor_list;
    }

private:
    box::Box m_box;
    NeighborList m_neighbor_list; //!< Stored neighbor list
};
}; }; // end namespace freud::locality

#endif // VORONOI_H
