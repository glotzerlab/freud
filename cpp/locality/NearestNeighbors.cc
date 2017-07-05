// Copyright (c) 2010-2016 The Regents of the University of Michigan
// This file is part of the Freud project, released under the BSD 3-Clause License.

#include <algorithm>
#include <stdexcept>
#include <complex>
#include <utility>
#include <vector>
#include <tbb/tbb.h>
#include <boost/math/special_functions/spherical_harmonic.hpp>

#include "NearestNeighbors.h"
#include "ScopedGILRelease.h"
#include "HOOMDMatrix.h"

using namespace std;
using namespace tbb;

/*! \file NearestNeighbors.h
  \brief Compute the hexatic order parameter for each particle
*/

namespace freud { namespace locality {

// stop using
NearestNeighbors::NearestNeighbors():
    m_box(box::Box()), m_rmax(0), m_num_neighbors(0), m_scale(0), m_strict_cut(false), m_num_points(0), m_num_ref(0),
    m_deficits()
    {
    m_lc = new locality::LinkCell();
    m_deficits = 0;
    }

NearestNeighbors::NearestNeighbors(float rmax,
                                   unsigned int num_neighbors,
                                   float scale,
                                   bool strict_cut):
    m_box(box::Box()), m_rmax(rmax), m_num_neighbors(num_neighbors), m_scale(scale), m_strict_cut(strict_cut), m_num_points(0),
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

    // TODO in the future, navigate the cell list directly rather than
    // repeatedly rebuilding after modifying r_cut
    typedef std::vector<std::tuple<size_t, size_t, float> > BondVector;
    typedef std::vector<BondVector> BondVectorVector;
    typedef tbb::enumerable_thread_specific<BondVectorVector> ThreadBondVector;
    ThreadBondVector bond_vectors;

    // find the nearest neighbors
    do
        {
        const float rmaxsq(m_rmax*m_rmax);
        // compute the cell list
        m_lc->compute(m_box, ref_pos, num_ref, pos, num_points, exclude_ii);
        const NeighborList *nlist(m_lc->getNeighborList());
        const size_t *neighbor_list(nlist->getNeighbors());
        bond_vectors.clear();

        m_deficits = 0;
        parallel_for(blocked_range<size_t>(0,num_ref),
            [=, &bond_vectors] (const blocked_range<size_t>& r)
            {
                size_t bond(nlist->find_first_index(r.begin()));
                ThreadBondVector::reference bond_vector_vectors(bond_vectors.local());
                bond_vector_vectors.emplace_back();
                BondVector &bond_vector(bond_vector_vectors.back());

                vector< pair<float, size_t> > neighbors;
                for(size_t i=r.begin(); i!=r.end(); ++i)
                {
                    // If we have found an incomplete set of neighbors, end now and rebuild
                    if((m_deficits > 0) && !(m_strict_cut))
                        break;
                    neighbors.clear();

                    //get cell point is in
                    vec3<float> posi = ref_pos[i];
                    for(; bond < nlist->getNumBonds() && neighbor_list[2*bond] == i; ++bond)
                    {
                        const size_t j(neighbor_list[2*bond + 1]);

                        //compute r between the two particles
                        vec3<float>rij = m_box.wrap(pos[j] - posi);
                        const float rsq = dot(rij, rij);

                        // adds all neighbors within rsq to list of possible neighbors
                        if (rsq < rmaxsq)
                        {
                            neighbors.emplace_back(rsq, j);
                        }
                    }

                    // Add to the deficit count if necessary
                    if((neighbors.size() < m_num_neighbors) && !(m_strict_cut))
                        m_deficits += (m_num_neighbors - neighbors.size());
                    else
                    {
                        // sort based on rsq
                        sort(neighbors.begin(), neighbors.end());
                        const unsigned int k_max = min((unsigned int) neighbors.size(), m_num_neighbors);
                        for (unsigned int k = 0; k < k_max; ++k)
                        {
                            bond_vector.emplace_back(i, neighbors[k].second, 1);
                        }
                    }
                }
            });

        // Increase m_rmax
        if((m_deficits > 0) && !(m_strict_cut))
            {
            m_rmax *= m_scale;
            // check if new r_max would be too large for the cell width
            vec3<float> L = m_box.getNearestPlaneDistance();
            bool too_wide =  m_rmax > L.x/2.0 || m_rmax > L.y/2.0;
            if (!m_box.is2D())
                {
                too_wide |=  m_rmax > L.z/2.0;
                }
            if (too_wide)
                {
                // throw runtime_warning("r_max has become too large to create a viable cell.");
                // for now print
                printf("r_max has become too large to create a viable cell. Returning neighbors found\n");
                m_deficits = 0;
                break;
                }
            m_lc->setCellWidth(m_rmax);
            }
        } while((m_deficits > 0) && !(m_strict_cut));

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
