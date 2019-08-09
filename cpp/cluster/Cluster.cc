// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <cassert>
#include <map>
#include <stdexcept>
#include <vector>

#include "Cluster.h"
#include "NeighborComputeFunctional.h"
#include "NeighborBond.h"
#include "dset/dset.h"

using namespace std;

/*! \file Cluster.cc
    \brief Routines for clustering points.
*/

namespace freud { namespace cluster {

Cluster::Cluster(float rcut) : m_rcut(rcut), m_num_particles(0), m_num_clusters(0)
{
    if (m_rcut < 0.0f)
        throw invalid_argument("Cluster requires that rcut must be non-negative.");
}

void Cluster::compute(const freud::locality::NeighborQuery* nq,
                      const freud::locality::NeighborList* nlist,
                      const vec3<float> *points,
                      unsigned int Np,
                      freud::locality::QueryArgs qargs)
{
    assert(points);
    assert(Np > 0);

    // reallocate the cluster_idx array if the size doesn't match the last one
    //std::cout << "Checking whether to resize" << std::endl;
    if (Np != m_num_particles)
    {
        //std::cout << "Resizing" << std::endl;
        m_cluster_idx.resize(Np);
    }
    //std::cout << "Upon compute (after resizing), the address of the cluster idx array shared_ptr<T> is " << (unsigned long) m_cluster_idx.m_data.get() << std::endl;
    //std::cout << "Upon compute (after resizing), the address of the cluster idx underlying T* is " << (unsigned long) ((*m_cluster_idx.m_data.get()).get()) << std::endl;

    m_num_particles = Np;
    DisjointSets dj(m_num_particles);

    freud::locality::loopOverNeighbors(
        nq, points, Np, qargs, nlist,
        [this, &dj](const freud::locality::NeighborBond& neighbor_bond) {
            // compute r between the two particles
            if (neighbor_bond.distance < m_rcut)
            {
                // merge the two sets using the disjoint set
                if (!dj.same(neighbor_bond.ref_id, neighbor_bond.id))
                {
                    dj.unite(neighbor_bond.ref_id, neighbor_bond.id);
                }
            }
        });

    // done looping over points. All clusters are now determined. Renumber them from zero to num_clusters-1.
    map<uint32_t, uint32_t> label_map;

    // go over every point
    uint32_t cur_set = 0;
    for (uint32_t i = 0; i < m_num_particles; i++)
    {
        uint32_t s = dj.find(i);

        // insert it into the mapping if we haven't seen this one yet
        if (label_map.count(s) == 0)
        {
            label_map[s] = cur_set;
            cur_set++;
        }

        // label this point in cluster_idx
        m_cluster_idx[i] = label_map[s];
    }

    // cur_set is now the number of clusters
    m_num_clusters = cur_set;

}

/*! \param keys Array of keys (1 per particle)
    Loops over all particles and adds them to a list of sets. Each set contains all the keys that are part of
    that cluster.

    Get the computed list with getClusterKeys().

    \note The length of keys is assumed to be the same length as the particles in the last call to
    computeClusters().
*/
void Cluster::computeClusterMembership(const unsigned int* keys)
{
    // clear the membership
    m_cluster_keys.resize(m_num_clusters);

    for (unsigned int i = 0; i < m_num_clusters; i++)
        m_cluster_keys[i].clear();

    // add members to the sets
    for (unsigned int i = 0; i < m_num_particles; i++)
    {
        unsigned int key = keys[i];
        unsigned int cluster = m_cluster_idx[i];
        m_cluster_keys[cluster].push_back(key);
    }
}

}; }; // end namespace freud::cluster
