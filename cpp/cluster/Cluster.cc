// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <map>
#include <stdexcept>
#include <vector>

#include "Cluster.h"

using namespace std;

/*! \file Cluster.cc
    \brief Routines for clustering points.
*/

namespace freud { namespace cluster {

/*! \param n Number of initial sets
*/
DisjointSet::DisjointSet(uint32_t n)
    : s(vector<uint32_t>(n)), rank(vector<uint32_t>(n, 0))
    {
    // initialize s
    for (uint32_t i = 0; i < n; i++)
        s[i] = i;
    }

/*! The two sets labeled \c a and \c b are merged
    \note Incorrect behavior if \c a == \c b or either are not set labels
*/
void DisjointSet::merge(const uint32_t a, const uint32_t b)
    {
    assert(a < s.size() && b < s.size()); // sanity check

    // if tree heights are equal, merge to a
    if (rank[a] == rank[b])
        {
        rank[a]++;
        s[b] = a;
        }
    else
        {
        // merge the shorter tree to the taller one
        if (rank[a] > rank[b])
            s[b] = a;
        else
            s[a] = b;
        }
    }

/*! \returns the set label that contains the element \c c
*/
uint32_t DisjointSet::find(const uint32_t c)
    {
    uint32_t r = c;

    // follow up to the root of the tree
    while (s[r] != r)
        r = s[r];

    // path compression
    uint32_t i = c;
    while (i != r)
        {
        uint32_t j = s[i];
        s[i] = r;
        i = j;
        }
    return r;
    }

Cluster::Cluster(float rcut)
    : m_rcut(rcut), m_num_particles(0), m_num_clusters(0)
    {
    if (m_rcut < 0.0f)
        throw invalid_argument("Cluster requires that rcut must be non-negative.");
    }

void Cluster::computeClusters(const box::Box& box,
                              const freud::locality::NeighborList *nlist,
                              const vec3<float> *points,
                              unsigned int Np)
    {
    assert(points);
    assert(Np > 0);

    nlist->validate(Np, Np);
    const size_t *neighbor_list(nlist->getNeighbors());

    // reallocate the cluster_idx array if the size doesn't match the last one
    if (Np != m_num_particles)
        m_cluster_idx = std::shared_ptr<unsigned int>(new unsigned int[Np], std::default_delete<unsigned int[]>());

    m_num_particles = Np;
    float rmaxsq = m_rcut * m_rcut;
    DisjointSet dj(m_num_particles);

    size_t bond(0);

    // for each point
    for (unsigned int i = 0; i < m_num_particles; i++)
        {
        // get the cell the point is in
        vec3<float> p = points[i];

        for(; bond < nlist->getNumBonds() && neighbor_list[2*bond] == i; ++bond)
            {
            const size_t j(neighbor_list[2*bond + 1]);
                {
                if (i != j)
                    {
                    // compute r between the two particles
                    vec3<float> delta = p - points[j];
                    delta = box.wrap(delta);

                    float rsq = dot(delta, delta);
                    if (rsq < rmaxsq)
                        {
                        // merge the two sets using the disjoint set
                        uint32_t a = dj.find(i);
                        uint32_t b = dj.find(j);
                        if (a != b)
                            dj.merge(a,b);
                        }
                    }
                }
            }
        }

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
        m_cluster_idx.get()[i] = label_map[s];
        }

    // cur_set is now the number of clusters
    m_num_clusters = cur_set;
    }

/*! \param keys Array of keys (1 per particle)
    Loops over all particles and adds them to a list of sets. Each set contains all the keys that are part of that cluster.

    Get the computed list with getClusterKeys().

    \note The length of keys is assumed to be the same length as the particles in the last call to computeClusters().
*/
void Cluster::computeClusterMembership(const unsigned int *keys)
    {
    // clear the membership
    m_cluster_keys.resize(m_num_clusters);

    for (unsigned int i = 0; i < m_num_clusters; i++)
        m_cluster_keys[i].clear();

    // add members to the sets
    for (unsigned int i = 0; i < m_num_particles; i++)
        {
        unsigned int key = keys[i];
        unsigned int cluster = m_cluster_idx.get()[i];
        m_cluster_keys[cluster].push_back(key);
        }
    }

}; }; // end namespace freud::cluster
