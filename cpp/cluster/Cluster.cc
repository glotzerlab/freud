// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <algorithm>
#include <cassert>
#include <map>
#include <numeric>
#include <stdexcept>

#include "Cluster.h"
#include "NeighborComputeFunctional.h"
#include "NeighborBond.h"
#include "dset/dset.h"

using namespace std;

/*! \file Cluster.cc
    \brief Routines for clustering points.
*/

namespace freud { namespace cluster {

Cluster::Cluster() : m_num_particles(0), m_num_clusters(0) {}

void Cluster::compute(const freud::locality::NeighborQuery* nq,
                      const freud::locality::NeighborList* nlist,
                      const vec3<float>* points, unsigned int Np,
                      freud::locality::QueryArgs qargs,
                      const unsigned int* keys)
{
    assert(points);
    assert(Np > 0);

    m_cluster_idx.prepare(Np);

    m_num_particles = Np;
    DisjointSets dj(m_num_particles);

    freud::locality::loopOverNeighbors(
        nq, points, Np, qargs, nlist,
        [this, &dj](const freud::locality::NeighborBond& neighbor_bond) {
            // merge the two sets using the disjoint set
            if (!dj.same(neighbor_bond.ref_id, neighbor_bond.id))
            {
                dj.unite(neighbor_bond.ref_id, neighbor_bond.id);
            }
        });

    // Done looping over points. All clusters are now determined.
    // Next, we renumber clusters from zero to num_clusters-1.
    // These new cluster indices are then sorted by cluster size from largest
    // to smallest, with equally-sized clusters sorted based on their minimum
    // particle index.
    map<size_t, size_t> label_map;
    map<size_t, size_t> label_counts;
    map<size_t, size_t> label_min_id;

    // Go over every point
    size_t cur_set = 0;
    for (size_t i = 0; i < m_num_particles; i++)
    {
        size_t s = dj.find(i);

        // Insert this cluster id into the mapping if we haven't seen it yet
        if (label_map.count(s) == 0)
        {
            label_map[s] = cur_set;
            label_min_id[cur_set] = m_num_particles;
            cur_set++;
        }

        // Increment the counter for this cluster label
        label_counts[label_map[s]]++;

        // Track the smallest particle index in this cluster
        label_min_id[label_map[s]] = std::min(label_min_id[label_map[s]], i);
    }

    // cur_set is now the total number of clusters
    m_num_clusters = cur_set;

    // Build vectors of counts/min_ids from the maps of label counts/min_ids
    vector<size_t> counts(m_num_clusters);
    vector<size_t> min_ids(m_num_clusters);
    for (size_t i = 0; i < m_num_clusters; i++)
    {
        counts[i] = label_counts[i];
        min_ids[i] = label_min_id[i];
    }

    // Get a permutation that reorders clusters, largest to smallest
    vector<size_t> cluster_reindex = sort_indexes_inverse(counts, min_ids);

    // Clear the cluster keys
    m_cluster_keys.resize(m_num_clusters);
    for (auto v : m_cluster_keys)
        v.clear();

    /* Loop over all particles, set their cluster ids and add them to a list of
     * sets. Each set contains all the keys that are part of that cluster. If
     * no keys are provided, the keys use particle ids. Get the computed list
     * with getClusterKeys().
    */
    for (size_t i = 0; i < m_num_particles; i++)
    {
        size_t s = dj.find(i);
        size_t cluster_idx = cluster_reindex[label_map[s]];
        m_cluster_idx[i] = cluster_idx;
        unsigned int key = i;
        if (keys != NULL)
            key = keys[i];
        m_cluster_keys[cluster_idx].push_back(key);
    }
}

// Returns inverse permutation of cluster indices, sorted from largest to smallest.
// Adapted from https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
std::vector<size_t> sort_indexes_inverse(const std::vector<size_t> &counts, const std::vector<size_t> &min_ids) {

    // Initialize original index locations
    std::vector<size_t> idx(counts.size());
    std::iota(idx.begin(), idx.end(), 0);

    // Sort indexes based on comparing values in counts, min_ids
    std::sort(idx.begin(), idx.end(), [&counts, &min_ids](size_t i1, size_t i2) {
        if (counts[i1] != counts[i2])
        {
            // If the counts are unequal, return the largest cluster first
            return counts[i1] > counts[i2];
        }
        else
        {
            // If the counts are equal, return the cluster with the smallest
            // particle id first
            return min_ids[i1] < min_ids[i2];
        }
    });

    // Invert the permutation
    std::vector<size_t> inv_idx(idx.size());
    for (size_t i = 0; i < idx.size(); i++)
        inv_idx[idx[i]] = i;
    return inv_idx;
}

}; }; // end namespace freud::cluster
