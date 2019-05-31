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
        for (unsigned int ridge = 0; ridge < n_ridges; ridge++){
            unsigned int i = expanded_ids[ridge_points[2*ridge]];
            unsigned int j = expanded_ids[ridge_points[2*ridge+1]];

            if (i >= N && j >= N)
                continue;

            bool exclude_ii = true;
            if (exclude_ii && i == j)
                continue;

            bool negone = false;
            for (int ridge_vert_id = ridge_vertex_indices[ridge]; ridge_vert_id
                < ridge_vertex_indices[ridge+1]; ++ridge_vert_id) {
                if (ridge_vertices[ridge_vert_id] == -1) {
                    negone = true;
                }
            }
            if(negone == false) {
                if (box.is2D()) {
                    weight = 1;
                } else {
                    weight = 1;
                }
            } else {
                weight = 0;
            }

        }


        // typedef tbb::enumerable_thread_specific< std::vector< std::pair<size_t, size_t> > > BondVector;
        // BondVector bonds;
        // tbb::parallel_for(tbb::blocked_range<size_t>(0, m_N),
        //         [&] (const tbb::blocked_range<size_t> &r)
        //         {
        //         BondVector::reference local_bonds(bonds.local());
        //         NeighborPoint np;
        //         for (size_t i(r.begin()); i != r.end(); ++i)
        //             {
        //             std::shared_ptr<NeighborQueryIterator> it = this->query(i);
        //             while (!it->end())
        //                 {
        //                 np = it->next();
        //                 // If we're excluding ii bonds, we have to check before adding.
        //                 if (!m_exclude_ii || i != np.ref_id)
        //                     {
        //                     // Swap ref_id and id order for backwards compatibility.
        //                     local_bonds.emplace_back(np.ref_id, i);
        //                     }
        //                 }
        //             // Remove the last item, which is just the terminal sentinel value.
        //             local_bonds.pop_back();
        //             }
        //         });

        // tbb::flattened2d<BondVector> flat_bonds = tbb::flatten2d(bonds);
        // std::vector< std::pair<size_t, size_t> > linear_bonds(flat_bonds.begin(), flat_bonds.end());
        // tbb::parallel_sort(linear_bonds.begin(), linear_bonds.end());

        // unsigned int num_bonds = linear_bonds.size();

        // NeighborList *nl = new NeighborList();
        // nl->resize(num_bonds);
        // nl->setNumBonds(num_bonds, m_neighbor_query->getNRef(), m_N);
        // size_t *neighbor_array(nl->getNeighbors());
        // float *neighbor_weights(nl->getWeights());

        // parallel_for(tbb::blocked_range<size_t>(0, num_bonds),
        //     [&] (const tbb::blocked_range<size_t> &r)
        //     {
        //     for (size_t bond(r.begin()); bond < r.end(); ++bond)
        //         {
        //         neighbor_array[2*bond] = linear_bonds[bond].first;
        //         neighbor_array[2*bond+1] = linear_bonds[bond].second;
        //         }
        //     });
        // memset((void*) neighbor_weights, 1, sizeof(float)*linear_bonds.size());

        // return nl;

    }

}; }; // end namespace freud::locality
