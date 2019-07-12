// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef VORONOI_H
#define VORONOI_H

#include "Box.h"
#include "VectorMath.h"
#include "NeighborList.h"

namespace freud { namespace locality {

struct NeighborBond
{
    NeighborBond() : id(0), ref_id(0), weight(0), distance(0) {}

    NeighborBond(unsigned int id, unsigned int ref_id, float d, float w) :
        id(id), ref_id(ref_id), distance(d), weight(w) {}

    unsigned int id;     //! The point id.
    unsigned int ref_id;     //! The reference point id.
    float weight;             //! The weight of this bond.
    float distance;           //! The distance bewteen the points.
};

class Voronoi
{
public:
    // default constructor
    Voronoi();

    void compute(const box::Box &box, const vec3<double>* vertices,
        const int* ridge_points, const int* ridge_vertices,
        unsigned int n_ridges, unsigned int N, const int* expanded_ids,
        const vec3<double>* expanded_points, const int* ridge_vertex_indices);

    NeighborList *getNeighborList()
    {
        return &m_neighbor_list;
    }

private:
    box::Box m_box;
    NeighborList m_neighbor_list; //!< Stored neighbor list

};
}; }; // end namespace freud::locality

#endif // VORONOI_H
