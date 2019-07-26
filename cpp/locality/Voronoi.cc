// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <cmath>
#include <stdexcept>
#include <tbb/tbb.h>
#include <tuple>
#include <vector>

#include "Voronoi.h"

#if defined _WIN32
#undef min // std::min clashes with a Windows header
#undef max // std::max clashes with a Windows header
#endif

/*! \file Voronoi.cc
    \brief Build a cell list from a set of points.
*/

namespace freud { namespace locality {

// Default constructor
Voronoi::Voronoi() {}

// A compare function used to sort NeighborBonds
bool compareNeighborPairs(const NeighborBond& n1, const NeighborBond& n2)
{
    if (n1.index_i != n2.index_i)
    {
        return n1.index_i < n2.index_i;
    }
    if (n1.index_j != n2.index_j)
    {
        return n1.index_j < n2.index_j;
    }
    return n1.weight < n2.weight;
}

typedef tbb::enumerable_thread_specific<std::vector<NeighborBond>> BondVector;

void add_valid_bonds(BondVector::reference local_bonds, unsigned int i, unsigned int expanded_i,
                     unsigned int j, unsigned int expanded_j, unsigned int N, float weight, float distance)
{
    // Make sure we only add bonds with real particles as the reference
    if (i < N && distance != 0)
    {
        NeighborBond nb_ij(expanded_i, expanded_j, weight, distance);
        local_bonds.emplace_back(nb_ij);
    }

    if (j < N && distance != 0)
    {
        NeighborBond nb_ji(expanded_j, expanded_i, weight, distance);
        local_bonds.emplace_back(nb_ji);
    }
}

// vertices is passed from scipy.spatial.Voronoi. It must keep in double precision.
// Any calculation related to vertices coords should also keep in double precision.
void Voronoi::compute(const box::Box& box, const vec3<double>* vertices, const int* ridge_points,
                      const int* ridge_vertices, unsigned int n_ridges, unsigned int N,
                      const int* expanded_ids, const vec3<double>* expanded_points,
                      const int* ridge_vertex_indices)
{
    m_box = box;

    // iterate over ridges in parallel
    BondVector bonds;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n_ridges), [&](const tbb::blocked_range<size_t>& r) {
        BondVector::reference local_bonds(bonds.local());
        for (size_t ridge(r.begin()); ridge != r.end(); ++ridge)
        {
            unsigned int i = ridge_points[2 * ridge];
            unsigned int j = ridge_points[2 * ridge + 1];
            float weight = 0;

            // compute distances bewteen two points
            vec3<double> rij(expanded_points[j] - expanded_points[i]);
            float distance = sqrt(dot(rij, rij));

            // Reject bonds between two image particles
            if (i >= N && j >= N)
                continue;

            // We DO allow bonds from a particle to its own image
            if (i == j)
                continue;

            // Bonds where a ridge goes to infinity (index-1) have weight 0
            bool weighted_bond = true;
            std::vector<vec3<double>> current_ridge_vertex;
            for (int ridge_vert_id = ridge_vertex_indices[ridge];
                 ridge_vert_id < ridge_vertex_indices[ridge + 1]; ++ridge_vert_id)
            {
                if (ridge_vertices[ridge_vert_id] == -1)
                {
                    add_valid_bonds(local_bonds, i, expanded_ids[i], j, expanded_ids[j], N, 0, distance);
                    weighted_bond = false;
                    break;
                }
                else
                {
                    current_ridge_vertex.push_back(vertices[ridge_vertices[ridge_vert_id]]);
                }
            }

            if (weighted_bond)
            {
                if (box.is2D())
                {
                    // 2D weight is the length of the ridge edge
                    vec3<double> v1 = current_ridge_vertex[0];
                    vec3<double> v2 = current_ridge_vertex[1];
                    // not necessary to have double precision in weight calculation
                    vec3<float> rij(box.wrap(v1 - v2));
                    weight = sqrt(dot(rij, rij));
                }
                else
                {
                    // 3D weight is the area of the ridge facet
                    // Create a vector of all vertices for this facet
                    std::vector<vec3<double>> vertex_coords;

                    for (std::vector<vec3<double>>::iterator ridge_vert_id = current_ridge_vertex.begin();
                         ridge_vert_id != current_ridge_vertex.end(); ++ridge_vert_id)
                    {
                        vec3<double> vert = *ridge_vert_id;
                        vertex_coords.push_back(vert);
                    }

                    // Code below is adapted from http://geomalgorithms.com/a01-_area.html
                    // Get a unit normal vector to the polygonal facet
                    // Every facet has at least 3 vertices
                    vec3<double> r01(vertex_coords[1] - vertex_coords[0]);
                    vec3<double> r12(vertex_coords[2] - vertex_coords[1]);
                    vec3<double> norm_vec = cross(r01, r12);
                    norm_vec /= sqrt(dot(norm_vec, norm_vec));

                    // Determine projection axis (x=0, y=1, z=2)
                    double c0_component = std::max(std::max(std::abs(norm_vec.x), std::abs(norm_vec.y)),
                                                   std::abs(norm_vec.z));
                    unsigned int c0 = 2;
                    if (c0_component == std::abs(norm_vec.x))
                    {
                        c0 = 0;
                    }
                    else if (c0_component == std::abs(norm_vec.y))
                    {
                        c0 = 1;
                    }

                    double projected_area = 0;
                    unsigned int n_verts = vertex_coords.size();
                    for (unsigned int step = 0; step < n_verts; step++)
                    {
                        unsigned int n1 = step % n_verts;
                        unsigned int n2 = (step + 1) % n_verts;
                        unsigned int n3 = (step + n_verts - 1) % n_verts;
                        switch (c0)
                        {
                        case 0:
                            projected_area
                                += vertex_coords[n1].y * (vertex_coords[n2].z - vertex_coords[n3].z);
                            break;
                        case 1:
                            projected_area
                                += vertex_coords[n1].z * (vertex_coords[n2].x - vertex_coords[n3].x);
                            break;
                        case 2:
                            projected_area
                                += vertex_coords[n1].x * (vertex_coords[n2].y - vertex_coords[n3].y);
                            break;
                        }
                    }
                    projected_area *= 0.5;

                    // Project back to get the true area (which is the weight)
                    weight = std::abs(projected_area / c0_component);
                }
                add_valid_bonds(local_bonds, i, expanded_ids[i], j, expanded_ids[j], N, weight, distance);
            }
        }
    });

    tbb::flattened2d<BondVector> flat_bonds = tbb::flatten2d(bonds);
    std::vector<NeighborBond> linear_bonds(flat_bonds.begin(), flat_bonds.end());
    tbb::parallel_sort(linear_bonds.begin(), linear_bonds.end(), compareNeighborPairs);

    unsigned int num_bonds = linear_bonds.size();

    m_neighbor_list.resize(num_bonds);
    m_neighbor_list.setNumBonds(num_bonds, N, N);
    size_t* neighbor_array(m_neighbor_list.getNeighbors());
    float* neighbor_weights(m_neighbor_list.getWeights());

    parallel_for(tbb::blocked_range<size_t>(0, num_bonds), [&](const tbb::blocked_range<size_t>& r) {
        for (size_t bond(r.begin()); bond < r.end(); ++bond)
        {
            neighbor_array[2 * bond] = linear_bonds[bond].index_i;
            neighbor_array[2 * bond + 1] = linear_bonds[bond].index_j;
            neighbor_weights[bond] = linear_bonds[bond].weight;
        }
    });
}

}; }; // end namespace freud::locality
