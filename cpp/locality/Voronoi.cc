// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <tbb/tbb.h>
#include <cmath>
#include <vector>

#include "Voronoi.h"
#include "NeighborBond.h"

/*! \file Voronoi.cc
    \brief Computes Voronoi neighbors for a set of points.
*/

namespace freud { namespace locality {

// Voronoi calculations should be kept in double precision.
void Voronoi::compute(const box::Box &box, const vec3<double>* points, unsigned int n_points)
    {
        m_polytopes.resize(n_points);
        m_volumes.prepare(n_points);

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

        for (size_t query_point_id = 0; query_point_id < n_points; query_point_id++) {
            container.put(query_point_id, points[query_point_id].x, points[query_point_id].y, points[query_point_id].z);
        }

        voro::voronoicell_neighbor cell;
        voro::c_loop_all_periodic voronoi_loop(container);
        std::vector<double> face_areas;
        std::vector<int> face_vertices;
        std::vector<int> neighbors;
        std::vector<double> normals;
        std::vector<double> vertices;
        std::vector<NeighborBond> bonds;

        if (voronoi_loop.start()) {
            do {
                container.compute_cell(cell, voronoi_loop);

                // Get id and position of current particle
                const int query_point_id(voronoi_loop.pid());
                vec3<double> ri(
                    voronoi_loop.x(),
                    voronoi_loop.y(),
                    voronoi_loop.z()
                );

                // Get Voronoi cell properties
                cell.face_areas(face_areas);
                cell.face_vertices(face_vertices);
                cell.neighbors(neighbors);
                cell.normals(normals);
                cell.vertices(ri.x, ri.y, ri.z, vertices);

                // Save polytope vertices in system coordinates
                std::vector<vec3<double>> vec3_vertices;
                auto vertex_iterator = vertices.begin();
                while (vertex_iterator != vertices.end()) {
                    double vert_x = *vertex_iterator;
                    vertex_iterator++;
                    double vert_y = *vertex_iterator;
                    vertex_iterator++;
                    double vert_z = *vertex_iterator;
                    vertex_iterator++;

                    // In 2D systems, only use vertices from the upper plane
                    // to prevent double-counting, and set z=0 manually
                    if (box.is2D()) {
                        if (vert_z < 0) {
                            continue;
                        }
                        vert_z = 0;
                    }
                    vec3_vertices.push_back(vec3<double>(vert_x, vert_y, vert_z));
                }
                m_polytopes[query_point_id] = vec3_vertices;

                // Save cell volume
                m_volumes[query_point_id] = cell.volume();

                size_t neighbor_counter(0);
                size_t face_vertices_index(0);
                for (auto neighbor_iterator = neighbors.begin(); neighbor_iterator != neighbors.end(); neighbor_iterator++) {
                    const int point_id = *neighbor_iterator;
                    const float weight(face_areas[neighbor_counter]);

                    // Get the normal to the current face
                    const vec3<double> normal(
                        normals[3*neighbor_counter],
                        normals[3*neighbor_counter+1],
                        normals[3*neighbor_counter+2]
                    );

                    // Find a vertex on the current face: this leverages the
                    // structure of face_vertices, which has a count of the
                    // number of vertices for a face followed by the
                    // corresponding vertex ids for that face. We use this
                    // structure later when incrementing face_vertices_index.
                    // face_vertices_index always points to the "vertex
                    // counter" element of face_vertices for the current face.

                    // Get the first vertex id on this face
                    const int vertex_id_on_face = face_vertices[face_vertices_index+1];

                    // Project the vertex vector onto the face normal to get a
                    // distance from ri to the face, then double it to get the
                    // distance to the neighbor particle
                    const vec3<double> rv(
                        vertices[3*vertex_id_on_face],
                        vertices[3*vertex_id_on_face+1],
                        vertices[3*vertex_id_on_face+2]
                    );
                    const vec3<double> riv(rv - ri);
                    const float distance(2*dot(riv, normal));


                    neighbor_counter++;
                    face_vertices_index += face_vertices[face_vertices_index] + 1;

                    // Ignore bonds in 2D systems that point up or down
                    if (box.is2D() && std::abs(normal.z) > 0)
                        continue;

                    bonds.push_back(NeighborBond(query_point_id, point_id, distance, weight));
                }

            } while (voronoi_loop.inc());
        }

        tbb::parallel_sort(bonds.begin(), bonds.end(),
                [](const NeighborBond& n1, const NeighborBond& n2) {
                    return n1.less_id_ref_weight(n2);
                });

        unsigned int num_bonds = bonds.size();

        m_neighbor_list->resize(num_bonds);
        m_neighbor_list->setNumBonds(num_bonds, n_points, n_points);

        util::forLoopWrapper(0, num_bonds, [=](size_t begin, size_t end) {
            for (size_t bond = begin; bond != end; ++bond)
            {
                m_neighbor_list->getNeighbors()(bond, 0) = bonds[bond].query_point_idx;
                m_neighbor_list->getNeighbors()(bond, 1) = bonds[bond].point_idx;
                m_neighbor_list->getDistances()[bond] = bonds[bond].distance;
                m_neighbor_list->getWeights()[bond] = bonds[bond].weight;
            }
        });

    }

}; }; // end namespace freud::locality
