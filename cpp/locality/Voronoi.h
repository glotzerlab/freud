// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef VORONOI_H
#define VORONOI_H

#include "VectorMath.h"
#include <vector>

namespace freud { namespace locality {

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
