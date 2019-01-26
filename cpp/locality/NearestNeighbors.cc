// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <algorithm>
#include <complex>
#include <stdexcept>
#include <tbb/tbb.h>
#include <utility>
#include <vector>

#include "NearestNeighbors.h"
#include "HOOMDMatrix.h"

using namespace std;
using namespace tbb;

/*! \file NearestNeighbors.h
  \brief Find the requested number of nearest neighbors.
*/

namespace freud { namespace locality {

// stop using
NearestNeighbors::NearestNeighbors():
    m_box(box::Box()), m_rmax(0), m_num_neighbors(0), m_strict_cut(false), m_num_points(0), m_num_ref(0),
    m_deficits()
    {
    m_lc = new locality::LinkCell();
    m_deficits = 0;
    }

NearestNeighbors::NearestNeighbors(float rmax,
                                   unsigned int num_neighbors,
                                   float scale,
                                   bool strict_cut):
    m_box(box::Box()), m_rmax(rmax), m_num_neighbors(num_neighbors), m_strict_cut(strict_cut), m_num_points(0),
    m_num_ref(0), m_deficits()
    {
    m_lc = new locality::LinkCell(m_box, m_rmax);
    m_deficits = 0;
    }

NearestNeighbors::~NearestNeighbors()
    {
    delete m_lc;
    }

void NearestNeighbors::setCutMode(const bool strict_cut)
    {
    m_strict_cut = strict_cut;
    }

void NearestNeighbors::compute(const box::Box& box,
                               const vec3<float> *ref_pos,
                               unsigned int num_ref,
                               const vec3<float> *pos,
                               unsigned int num_points,
                               bool exclude_ii)
    {
    m_box = box;
    m_neighbor_list.resize(num_ref*m_num_neighbors);

    typedef std::vector<std::tuple<size_t, size_t, float> > BondVector;
    typedef std::vector<BondVector> BondVectorVector;
    typedef tbb::enumerable_thread_specific<BondVectorVector> ThreadBondVector;
    ThreadBondVector bond_vectors;

    m_lc->computeCellList(m_box, pos, num_points);
    const float rcutsq(m_lc->getCellWidth()*m_lc->getCellWidth());

    // find the nearest neighbors
    parallel_for(blocked_range<size_t>(0,num_ref),
        [=, &bond_vectors] (const blocked_range<size_t>& r)
            {
            ThreadBondVector::reference bond_vector_vectors(bond_vectors.local());
            bond_vector_vectors.emplace_back();
            BondVector &bond_vector(bond_vector_vectors.back());
            const Index3D &indexer(m_lc->getCellIndexer());
            const unsigned int max_cell_distance_2d(min(indexer.getW(), indexer.getH()));
            const unsigned int max_cell_distance_3d(min(max_cell_distance_2d, indexer.getD()));
            const unsigned int max_cell_distance(m_box.is2D()? max_cell_distance_2d: max_cell_distance_3d);

            // neighbors is the set of bonds we find that are within the cutoff radius
            vector<pair<float, size_t> > neighbors;
            // backup_neighbors is the set of bonds that are outside of
            // the cutoff radius (but should be used next time we increase
            // the range of cells we are looking within)
            vector<pair<float, size_t> > backup_neighbors;

            for(size_t i(r.begin()); i != r.end(); ++i)
                {
                const vec3<float> ref_point(ref_pos[i]);
                // look for cells in [min_iter_distance, max_iter_distance)
                unsigned int min_iter_distance(0), max_iter_distance(2);
                neighbors.clear();
                backup_neighbors.clear();
                // hit_max_distance should be updated each time we change
                // the maximum distance to make sure we don't go over half
                // the box length
                bool hit_max_distance(false);

                do
                    {
                    neighbors.insert(neighbors.end(), backup_neighbors.begin(), backup_neighbors.end());
                    backup_neighbors.clear();
                    const vec3<unsigned int> refCell(m_lc->getCellCoord(ref_pos[i]));

                    for(IteratorCellShell neigh_cell_iter(min_iter_distance, m_box.is2D());
                        neigh_cell_iter != IteratorCellShell(max_iter_distance, m_box.is2D()); ++neigh_cell_iter)
                        {
                        const vec3<int> neighbor_cell_delta(*neigh_cell_iter);
                        if(2*neighbor_cell_delta.x + 1 > (int) indexer.getW())
                            continue;
                        else if(2*neighbor_cell_delta.y + 1 > (int) indexer.getH())
                            continue;
                        else if(2*neighbor_cell_delta.z + 1 > (int) indexer.getD())
                            continue;

                        vec3<int> neighborCellCoords(refCell.x, refCell.y, refCell.z);
                        neighborCellCoords += neighbor_cell_delta;
                        if(neighborCellCoords.x < 0)
                            neighborCellCoords.x += indexer.getW();
                        neighborCellCoords.x %= indexer.getW();
                        if(neighborCellCoords.y < 0)
                            neighborCellCoords.y += indexer.getH();
                        neighborCellCoords.y %= indexer.getH();
                        if(neighborCellCoords.z < 0)
                            neighborCellCoords.z += indexer.getD();
                        neighborCellCoords.z %= indexer.getD();

                        const size_t neighborCellIndex(indexer(neighborCellCoords.x, neighborCellCoords.y, neighborCellCoords.z));

                        // iterate over the particles in that cell
                        locality::LinkCell::iteratorcell it = m_lc->itercell(neighborCellIndex);
                        for (unsigned int j = it.next(); !it.atEnd(); j=it.next())
                            {
                            if(exclude_ii && i == j)
                                continue;

                            const vec3<float> rij(m_box.wrap(pos[j] - ref_point));
                            const float rsq(dot(rij, rij));

                            if(rsq < (max_iter_distance - 1)*(max_iter_distance - 1)*rcutsq)
                                neighbors.emplace_back(rsq, j);
                            else
                                backup_neighbors.emplace_back(rsq, j);
                            }
                        }

                    hit_max_distance = 2*max_iter_distance > max_cell_distance;
                    min_iter_distance = max_iter_distance;
                    ++max_iter_distance;
                    } while((neighbors.size() < m_num_neighbors) && !m_strict_cut && !hit_max_distance);

                // if we looked at the maximum cell range, add the backup
                // particles that we found
                if(!m_strict_cut && hit_max_distance)
                    neighbors.insert(neighbors.end(), backup_neighbors.begin(), backup_neighbors.end());
                sort(neighbors.begin(), neighbors.end());
                const unsigned int k_max = min((unsigned int) neighbors.size(), m_num_neighbors);
                for (unsigned int k = 0; k < k_max; ++k)
                    {
                    bond_vector.emplace_back(i, neighbors[k].second, 1);
                    }
                }
            });

    // Sort neighbors by particle i index
    tbb::flattened2d<ThreadBondVector> flat_bond_vector_groups = tbb::flatten2d(bond_vectors);
    BondVectorVector bond_vector_groups(flat_bond_vector_groups.begin(), flat_bond_vector_groups.end());
    tbb::parallel_sort(bond_vector_groups.begin(), bond_vector_groups.end(), compareFirstNeighborPairs);

    unsigned int num_bonds(0);
    for(BondVectorVector::const_iterator iter(bond_vector_groups.begin());
        iter != bond_vector_groups.end(); ++iter)
        num_bonds += iter->size();

    m_neighbor_list.setNumBonds(num_bonds, num_ref, num_points);

    size_t *neighbor_array(m_neighbor_list.getNeighbors());
    float *neighbor_weights(m_neighbor_list.getWeights());

    // build nlist structure
    parallel_for(blocked_range<size_t>(0, bond_vector_groups.size()),
        [=, &bond_vector_groups] (const blocked_range<size_t> &r)
            {
            size_t bond(0);
            for(size_t group(0); group < r.begin(); ++group)
                bond += bond_vector_groups[group].size();

            for(size_t group(r.begin()); group < r.end(); ++group)
                {
                const BondVector &vec(bond_vector_groups[group]);
                for(BondVector::const_iterator iter(vec.begin());
                    iter != vec.end(); ++iter, ++bond)
                    {
                    std::tie(neighbor_array[2*bond], neighbor_array[2*bond + 1],
                        neighbor_weights[bond]) = *iter;
                    }
                }
            });

    // save the last computed number of particles
    m_num_ref = num_ref;
    m_num_points = num_points;
    }

}; }; // end namespace freud::locality
