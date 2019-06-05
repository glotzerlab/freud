// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef VORONOI_H
#define VORONOI_H

#include "VectorMath.h"
#include <vector>

namespace freud { namespace locality {

struct NeighborBond {
    NeighborBond() : index_i(0), index_j(0), weight(0) {}

    NeighborBond(unsigned int index_i, unsigned int index_j, float w) :
        index_i(index_i), index_j(index_j), weight(w) {}

    //! Equality checks both i, j pair and weight.
    bool operator== (const NeighborBond &n)
        {
        return (index_i == n.index_i) && (index_j == n.index_j) && (weight == n.weight);
        }

    //! Default comparator of points is by weight.
    /*! This form of comparison allows easy sorting of nearest neighbors by
     *  weight.
     */
    bool operator< (const NeighborBond &n) const
        {
        return weight < n.weight;
        }

    unsigned int index_i;     //! The point id.
    unsigned int index_j;     //! The reference point id.
    float weight;             //! The weight of this bond.
};


class Voronoi
    {
    public:
        // Null constructor
        Voronoi();

        void print_hello();

        // void compute(const vec3<double>* vertices, const std::vector<int>* ridge_points, const std::vector<int>* ridge_vertices);
        void compute(const box::Box &box, const vec3<double>* vertices,
            const int* ridge_points, const int* ridge_vertices,
            unsigned int n_ridges, unsigned int N, const int* expanded_ids,
            const int* ridge_vertex_indices);

    private:
        box::Box m_box;
    };
}; }; // end namespace freud::locality

#endif // VORONOI_H
