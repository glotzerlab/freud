// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <tbb/tbb.h>
#include <tuple>
#include <cmath>
#include <memory>
#include <vector>

#include "Box.h"
#include "NeighborList.h"
#include "Voronoi.h"
#include "NeighborQuery.h"

using namespace std;
using namespace tbb;

/*! \file Voronoi.cc
    \brief Build a cell list from a set of points.
*/

namespace freud { namespace locality {

// Default constructor
Voronoi::Voronoi()
    {}

void Voronoi::print_hello()
    {
        std::cout << "hello Yezhi" << std::endl;
    }

void Voronoi::compute(const box::Box &box, const vec3<double>* vertices,
    const int* ridge_points, const int* ridge_vertices, unsigned int n_ridges,
    unsigned int N, const int* expanded_ids, const int* ridge_vertex_indices)
    {
        m_box = box;

        float weight = 0;
        // iterate over ridges in parallel
        typedef tbb::enumerable_thread_specific< std::vector<NeighborBond> > BondVector;
        BondVector bonds;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n_ridges), [&] (const tbb::blocked_range<size_t> &r) {
             BondVector::reference local_bonds(bonds.local());
            for (size_t ridge(r.begin()); ridge != r.end(); ++ridge) {
                unsigned int i = expanded_ids[ridge_points[2*ridge]];
                unsigned int j = expanded_ids[ridge_points[2*ridge+1]];

                if (i >= N && j >= N)
                    continue;

                bool exclude_ii = true;
                if (exclude_ii && i == j)
                    continue;

                // Reject bonds where a ridge goes to infinity (index -1)
                bool valid_bond = true;
                for (int ridge_vert_id = ridge_vertex_indices[ridge]; ridge_vert_id < ridge_vertex_indices[ridge+1]; ++ridge_vert_id) {
                    if (ridge_vertices[ridge_vert_id] == -1) {
                        valid_bond = false;
                    }
                }
                if (valid_bond) {
                    if (box.is2D()) {
                        cout << "This is 2D" << endl;
                        // 2D weight is the length of the ridge edge
                        vec3<double> rij(vertices[ridge_vertices[2*ridge]] - vertices[ridge_vertices[2*ridge+1]]);
                        weight = sqrt(dot(rij, rij));
                    } else {
                        cout << "This is 3D" << endl;
                        // 3D weight is the area of the ridge facet
                        // Create a vector of all vertices for this facet
                        vector< vec3<double> > vertex_coords;
                        for (int ridge_vert_id = ridge_vertex_indices[ridge];
                            ridge_vert_id < ridge_vertex_indices[ridge+1];
                            ++ridge_vert_id)
                        {
                            vertex_coords.push_back(vertices[ridge_vertices[ridge_vert_id]]);
                        }

                        cout << "There are " << vertex_coords.size() << " vertices in this ridge." << endl;

                        // Code below is adapted from http://geomalgorithms.com/a01-_area.html
                        // Get a unit normal vector to the polygonal facet
                        // Every facet has at least 3 vertices
                        vec3<double> r01(vertex_coords[1] - vertex_coords[0]);
                        vec3<double> r12(vertex_coords[2] - vertex_coords[1]);
                        vec3<double> norm_vec = cross(r01, r12);
                        norm_vec /= sqrt(dot(norm_vec, norm_vec));

                        // Determine projection axis (x=0, y=1, z=2)
                        double c0_component = std::max(std::max(abs(norm_vec.x), abs(norm_vec.y)), abs(norm_vec.z));
                        int c0 = 2;
                        if (c0_component == abs(norm_vec.x)) {
                            c0 = 0;
                        } else if (c0_component == abs(norm_vec.y)) {
                            c0 = 1;
                        }

                        double projected_area = 0;
                        int n_verts = vertex_coords.size();
                        for (int step = 0; step < n_verts; step++)
                        {
                            int n1 = step % n_verts;
                            int n2 = (step + 1) % n_verts;
                            int n3 = (step - 1 + n_verts) % n_verts;
                            cout << "n1 = " << n1 << ", n2 = " << n2 << ", n3 = " << n3 << endl;
                            switch(c0)
                            {
                                case 0:
                                    projected_area += abs(vertex_coords[n1].y * (vertex_coords[n2].z - vertex_coords[n3].z));
                                    break;
                                case 1:
                                    cout << "area: " << abs(vertex_coords[n1].z * (vertex_coords[n2].x - vertex_coords[n3].x)) << endl;
                                    projected_area += abs(vertex_coords[n1].z * (vertex_coords[n2].x - vertex_coords[n3].x));
                                    break;
                                case 2:
                                    cout << "area: " << (vertex_coords[n1].x * (vertex_coords[n2].y - vertex_coords[n3].y)) << endl;
                                    projected_area += abs(vertex_coords[n1].x * (vertex_coords[n2].y - vertex_coords[n3].y));
                                    break;
                            }
                        }
                        projected_area *= 0.5;

                        // Project back to get the true area (which is the weight)
                        weight = projected_area / abs(c0_component);
                        cout << "Projected area: " << projected_area << endl;
                        cout << "c0: " << c0 << ", component: " << c0_component << endl;
                    }
                    cout << "i = " << i << ", j = " << j << ", weight = " << weight << endl;
                    NeighborBond nb(i, j, weight);
                    local_bonds.emplace_back(nb);
                }
            }

        });

        tbb::flattened2d<BondVector> flat_bonds = tbb::flatten2d(bonds);
        std::vector<NeighborBond> linear_bonds(flat_bonds.begin(), flat_bonds.end());
        tbb::parallel_sort(linear_bonds.begin(), linear_bonds.end());

        unsigned int num_bonds = linear_bonds.size();

        NeighborList *nl = new NeighborList();
        nl->resize(num_bonds);
        nl->setNumBonds(num_bonds, N, N);
        size_t *neighbor_array(nl->getNeighbors());
        float *neighbor_weights(nl->getWeights());

        parallel_for(tbb::blocked_range<size_t>(0, num_bonds),
            [&] (const tbb::blocked_range<size_t> &r) {
            for (size_t bond(r.begin()); bond < r.end(); ++bond) {
                neighbor_array[2*bond] = linear_bonds[bond].index_i;
                neighbor_array[2*bond+1] = linear_bonds[bond].index_j;
                neighbor_weights[bond] = linear_bonds[bond].weight;
            }
        });

    }

}; }; // end namespace freud::locality
