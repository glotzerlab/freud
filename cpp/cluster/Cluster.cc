// Copyright (c) 2010-2016 The Regents of the University of Michigan
// This file is part of the Freud project, released under the BSD 3-Clause License.

#include "Cluster.h"

#include <stdexcept>
#include <vector>
#include <map>

using namespace std;

/*! \file Cluster.cc
    \brief Routines for clustering points
*/

namespace freud { namespace cluster {

/*! \param n Number of initial sets
*/
DisjointSet::DisjointSet(uint32_t n)
    {
    s = vector<uint32_t>(n);
    rank = vector<uint32_t>(n, 0);

    // initialize s
    for (uint32_t i = 0; i < n; i++)
        s[i] = i;
    }

/*! The two sets labelled \c a and \c b are merged
    \note Incorrect behaivior if \c a == \c b or either are not set labels
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

Cluster::Cluster(const box::Box& box, float rcut)
    : m_box(box), m_rcut(rcut), m_num_particles(0)
    {
    if (m_rcut < 0.0f)
        throw invalid_argument("rcut must be positive");
    }

// void Cluster::computeClusters(const float3 *points,
//                               unsigned int Np)
void Cluster::computeClusters(const freud::locality::NeighborList *nlist,
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
                    // float dx = float(p.x - points[j].x);
                    // float dy = float(p.y - points[j].y);
                    // float dz = float(p.z - points[j].z);
                    delta = m_box.wrap(delta);

                    // float rsq = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
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

// void Cluster::computeClustersPy(boost::python::numeric::array points)
//     {
//     // validate input type and rank
//     num_util::check_type(points, NPY_FLOAT);
//     num_util::check_rank(points, 2);

//     // validate that the 2nd dimension is only 3
//     num_util::check_dim(points, 1, 3);
//     unsigned int Np = num_util::shape(points)[0];

//     // get the raw data pointers and compute the cell list
//     // float3* points_raw = (float3*) num_util::data(points);
//     vec3<float>* points_raw = (vec3<float>*) num_util::data(points);

//     computeClusters(points_raw, Np);
//     }

/*! \param keys Array of keys (1 per particle)

    Loops overa all particles and adds them to a list of sets. Each set contains all the keys that are part of that
    cluster.

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
        m_cluster_keys[cluster].insert(key);
        }
    }

// /*! \param keys numpy array of uints, one for each particle.

//     Each particle is given a key (more than one particle can share the same key). getClusterKeys determines which keys
//     are present in each cluster. It returns a list of lists. List i in the return value is the list of keys that
//     are present in cluster i.
// */
// void Cluster::computeClusterMembershipPy(boost::python::numeric::array keys)
//     {
//     // validate input type and rank
//     num_util::check_type(keys, NPY_UINT32);
//     num_util::check_rank(keys, 1);

//     // Check that there is one key per point
//     unsigned int Np = num_util::shape(keys)[0];

//     if (!(Np == m_num_particles))
//         throw invalid_argument("Number of keys must be equal to the number of particles last handled by compute()");

//     // get the raw data pointer to the keys
//     unsigned int* keys_raw = (unsigned int *)num_util::data(keys);
//     computeClusterMembership(keys_raw);
//     }

// /*! Converts m_cluster_keys into a python list of lists
// */
// boost::python::object Cluster::getClusterKeysPy()
//     {
//     boost::python::list cluster_keys_py;
//     for (unsigned int i = 0; i < m_cluster_keys.size(); i++)
//         {
//         boost::python::list members;
//         set<unsigned int>::iterator k;
//         for (k = m_cluster_keys[i].begin(); k != m_cluster_keys[i].end(); ++k)
//             members.append(*k);

//         cluster_keys_py.append(members);
//         }
//     return cluster_keys_py;
//     }

// void export_Cluster()
//     {
//     class_<Cluster>("Cluster", init<box::Box&, float>())
//         .def("getBox", &Cluster::getBox, return_internal_reference<>())
//         .def("computeClusters", &Cluster::computeClustersPy)
//         .def("getNumClusters", &Cluster::getNumClusters)
//         .def("getClusterIdx", &Cluster::getClusterIdxPy)
//         .def("computeClusterMembership", &Cluster::computeClusterMembershipPy)
//         .def("getClusterKeys", &Cluster::getClusterKeysPy)
//         ;
//     }

}; }; // end namespace freud::cluster
