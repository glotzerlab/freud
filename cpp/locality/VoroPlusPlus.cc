// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <stdexcept>
#include <tbb/tbb.h>
#include <tuple>
#include <cmath>
#include <vector>

#include "VoroPlusPlus.h"

#if defined _WIN32
#undef min // std::min clashes with a Windows header
#undef max // std::max clashes with a Windows header
#endif

/*! \file VoroPlusPlus.cc
    \brief Build a cell list from a set of points.
*/

namespace freud { namespace locality {

// Default constructor
VoroPlusPlus::VoroPlusPlus()
{}

// A compare function used to sort VoroPlusPlusBonds
bool compareNeighborPairs(const VoroPlusPlusBond &n1, const VoroPlusPlusBond &n2)
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

typedef tbb::enumerable_thread_specific< std::vector<VoroPlusPlusBond> > BondVector;
typedef std::vector<VoroPlusPlusBond> SerialBondVector;

// Voronoi calculations should be kept in double precision.
void VoroPlusPlus::compute(const box::Box &box, const vec3<double>* points, unsigned int N)
    {
        m_polytopes.clear();
        m_volumes.clear();

        vec3<float> boxLatticeVectors[3];
        boxLatticeVectors[0] = box.getLatticeVector(0);
        boxLatticeVectors[1] = box.getLatticeVector(1);
        if (box.is2D()) {
            boxLatticeVectors[2] = vec3<float>(0, 0, 1);
        } else {
            boxLatticeVectors[2] = box.getLatticeVector(2);
        }
        voro::container_periodic container(
            boxLatticeVectors[0].x,
            boxLatticeVectors[1].x,
            boxLatticeVectors[1].y,
            boxLatticeVectors[2].x,
            boxLatticeVectors[2].y,
            boxLatticeVectors[2].z,
            3, 3, 3, 3
        );

        for (size_t pid = 0; pid < N; pid++) {
            container.put(pid, points[pid].x, points[pid].y, points[pid].z);
        }

        voro::voronoicell_neighbor cell;
        voro::c_loop_all_periodic voronoi_loop(container);
        std::vector<double> face_areas;
        std::vector<int> face_orders;
        std::vector<int> face_vertices;
        std::vector<int> neighbors;
        std::vector<double> normals;
        std::vector<double> vertices;
        SerialBondVector bonds;
        bool print_loud = false;

        if (voronoi_loop.start()) {
            do {
                container.compute_cell(cell, voronoi_loop);

                // Get id and position of current particle
                int pid(voronoi_loop.pid());
                vec3<double> ri(
                    voronoi_loop.x(),
                    voronoi_loop.y(),
                    voronoi_loop.z()
                );

                // Get Voronoi cell properties
                cell.face_areas(face_areas);
                cell.face_orders(face_orders);
                cell.face_vertices(face_vertices);
                cell.neighbors(neighbors);
                cell.normals(normals);
                cell.vertices(vertices);

                // Save polytope vertices
                //TODO: Only use upper plane (z > 0) vertices if the box is 2D
                // and set z=0 manually
                std::vector<vec3<double>> vec3_vertices;
                auto vertex_iterator = vertices.begin();
                while (vertex_iterator != vertices.end()) {
                    double x = *vertex_iterator;
                    vertex_iterator++;
                    double y = *vertex_iterator;
                    vertex_iterator++;
                    double z = *vertex_iterator;
                    vertex_iterator++;
                    vec3_vertices.push_back(vec3<double>(x, y, z));
                }
                m_polytopes.push_back(vec3_vertices);

                // Save cell volume
                m_volumes.push_back(cell.volume());

                size_t neighbor_counter(0);
                for (auto neighbor_iterator = neighbors.begin(); neighbor_iterator != neighbors.end(); neighbor_iterator++) {
                    int neighbor_id = *neighbor_iterator;
                    float weight(face_areas[neighbor_counter]);

                    // Get the normal to the current face
                    vec3<double> normal(
                        normals[3*neighbor_counter],
                        normals[3*neighbor_counter+1],
                        normals[3*neighbor_counter+2]
                    );

                    // Find a vertex on the current face:
                    //
                    // Leverages structure of face_vertices, which has a count
                    // of the number of vertices for that face followed by the
                    // corresponding vertex ids for each face.
                    //
                    // First, skip through the previous faces
                    int face_vertices_index = 0;
                    for (size_t face_counter = 0; face_counter < neighbor_counter; face_counter++) {
                        face_vertices_index += face_vertices[face_vertices_index] + 1;
                    }

                    // Get the first vertex id on this face
                    int vertex_id_on_face = face_vertices[face_vertices_index+1];

                    // Project the vertex vector onto the face normal to get a
                    // distance from ri to the face, then double it to get the
                    // distance to the neighbor particle
                    vec3<double> rv(
                        vertices[3*vertex_id_on_face],
                        vertices[3*vertex_id_on_face+1],
                        vertices[3*vertex_id_on_face+2]
                    );
                    vec3<double> riv(rv - ri);
                    float dist(2*dot(riv, normal));


                    neighbor_counter++;
                    printf("Bond from %i to %i, weight %f, distance %f, normal (%f, %f, %f)\n", pid, neighbor_id, weight, dist, normal.x, normal.y, normal.z);
                    printf("Vertex %i on face, ri (%f, %f, %f), rv (%f, %f, %f)\n", vertex_id_on_face, ri.x, ri.y, ri.z, rv.x, rv.y, rv.z);
                    bonds.push_back(VoroPlusPlusBond(pid, neighbor_id, weight, dist));
                }

                if (print_loud) {
                    // Print id and position
                    printf("\n\npid, xyz: ");
                    printf("%i (%f, %f, %f)\n", pid, ri.x, ri.y, ri.z);

                    // Print normals
                    printf("Normals: ");
                    for (std::vector<double>::iterator nn = normals.begin(); nn != normals.end(); nn++) {
                        printf("%f ", *nn);
                    }
                    printf("\n");

                    // Print neighbors
                    printf("Neighbors: ");
                    for (std::vector<int>::iterator nn = neighbors.begin(); nn != neighbors.end(); nn++) {
                        printf("%i ", *nn);
                    }
                    printf("\n");

                    // Print face areas
                    printf("Face areas: ");
                    for (std::vector<double>::iterator fa = face_areas.begin(); fa != face_areas.end(); fa++) {
                        printf("%f ", *fa);
                    }
                    printf("\n");

                    // Print vertices
                    printf("Vertices: ");
                    for (std::vector<double>::iterator vv = vertices.begin(); vv != vertices.end(); vv++) {
                        printf("%f ", *vv);
                    }
                    printf("\n");
                }
            } while (voronoi_loop.inc());
        }

        tbb::parallel_sort(bonds.begin(), bonds.end(), compareNeighborPairs);

        unsigned int num_bonds = bonds.size();

        m_neighbor_list.resize(num_bonds);
        m_neighbor_list.setNumBonds(num_bonds, N, N);
        size_t *neighbor_array(m_neighbor_list.getNeighbors());
        float *neighbor_weights(m_neighbor_list.getWeights());

        parallel_for(tbb::blocked_range<size_t>(0, num_bonds),
            [&] (const tbb::blocked_range<size_t> &r) {
            for (size_t bond(r.begin()); bond < r.end(); ++bond)
            {
                neighbor_array[2*bond] = bonds[bond].index_i;
                neighbor_array[2*bond+1] = bonds[bond].index_j;
                neighbor_weights[bond] = bonds[bond].weight;
            }
        });

        /*
        // iterate over ridges in parallel
        BondVector bonds;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n_ridges), [&] (const tbb::blocked_range<size_t> &r) {
             BondVector::reference local_bonds(bonds.local());
            for (size_t ridge(r.begin()); ridge != r.end(); ++ridge) {
                unsigned int i = ridge_points[2*ridge];
                unsigned int j = ridge_points[2*ridge+1];
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
                std::vector< vec3<double> > current_ridge_vertex;
                for (int ridge_vert_id = ridge_vertex_indices[ridge]; ridge_vert_id < ridge_vertex_indices[ridge+1]; ++ridge_vert_id)
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
                        std::vector< vec3<double> > vertex_coords;

                        for (std::vector< vec3<double> >::iterator ridge_vert_id = current_ridge_vertex.begin();
                            ridge_vert_id != current_ridge_vertex.end();
                            ++ridge_vert_id)
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
                        double c0_component = std::max(std::max(std::abs(norm_vec.x), std::abs(norm_vec.y)), std::abs(norm_vec.z));
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
                            unsigned int n3 = (step + n_verts - 1 ) % n_verts;
                            switch(c0)
                            {
                                case 0:
                                    projected_area += vertex_coords[n1].y * (vertex_coords[n2].z - vertex_coords[n3].z);
                                    break;
                                case 1:
                                    projected_area += vertex_coords[n1].z * (vertex_coords[n2].x - vertex_coords[n3].x);
                                    break;
                                case 2:
                                    projected_area += vertex_coords[n1].x * (vertex_coords[n2].y - vertex_coords[n3].y);
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
        std::vector<VoroPlusPlusBond> linear_bonds(flat_bonds.begin(), flat_bonds.end());
        tbb::parallel_sort(linear_bonds.begin(), linear_bonds.end(), compareNeighborPairs);

        unsigned int num_bonds = linear_bonds.size();

        m_neighbor_list.resize(num_bonds);
        m_neighbor_list.setNumBonds(num_bonds, N, N);
        size_t *neighbor_array(m_neighbor_list.getNeighbors());
        float *neighbor_weights(m_neighbor_list.getWeights());

        parallel_for(tbb::blocked_range<size_t>(0, num_bonds),
            [&] (const tbb::blocked_range<size_t> &r) {
            for (size_t bond(r.begin()); bond < r.end(); ++bond)
            {
                neighbor_array[2*bond] = linear_bonds[bond].index_i;
                neighbor_array[2*bond+1] = linear_bonds[bond].index_j;
                neighbor_weights[bond] = linear_bonds[bond].weight;
            }
        });
        */

    }

}; }; // end namespace freud::locality
