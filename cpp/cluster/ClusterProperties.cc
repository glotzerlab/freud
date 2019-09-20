// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

#include "ClusterProperties.h"

/*! \file ClusterProperties.cc
    \brief Routines for computing properties of point clusters.
*/

namespace freud { namespace cluster {

ClusterProperties::ClusterProperties() : m_num_clusters(0) {}

/*! \param box Box containing the particles
    \param points Positions of the particles making up the clusters
    \param cluster_idx Index of which cluster each point belongs to
    \param Np Number of particles (length of \a points and \a cluster_idx)

    computeClusterProperties loops over all points in the given array and determines the center of mass of the
   cluster as well as the G tensor. These can be accessed after the call to compute with getClusterCOM() and
   getClusterG().
*/
void ClusterProperties::computeProperties(const box::Box& box, const vec3<float>* points,
                                          const unsigned int* cluster_idx, unsigned int Np)
{
    // determine the number of clusters
    const unsigned int* max_cluster_id = std::max_element(cluster_idx, cluster_idx + Np);
    m_num_clusters = *max_cluster_id + 1;

    // allocate memory for the cluster properties and temporary arrays
    // initialize arrays to 0
    m_cluster_com.prepare(m_num_clusters);
    m_cluster_G.prepare({m_num_clusters, 3, 3});
    m_cluster_size.prepare(m_num_clusters);

    // ref_particle is the first particle found in a cluster, it is used as a
    // reference to compute the COM in relation to, for handling of the
    // periodic boundary conditions
    std::vector<vec3<float>> ref_pos(m_num_clusters, vec3<float>(0.0f, 0.0f, 0.0f));
    // determine if we have seen this cluster before or not (used to initialize ref_pos)
    std::vector<bool> cluster_seen(m_num_clusters, false);

    // Start by determining the center of mass of each cluster. Since we are
    // given an array of particles, the easiest way to do this is to loop over
    // all particles and add the appropriate information to m_cluster_com as
    // we go.
    for (unsigned int i = 0; i < Np; i++)
    {
        unsigned int c = cluster_idx[i];
        vec3<float> pos = points[i];

        // the first time we see the cluster, mark this point as the reference position
        if (!cluster_seen[c])
        {
            ref_pos[c] = pos;
            cluster_seen[c] = true;
        }

        // To compute the COM in periodic boundary conditions, compute all
        // reference vectors as wrapped vectors relative to ref_pos. When we
        // are done, we can add the computed COM to ref_pos to get the COM in
        // the space frame.
        vec3<float> delta = pos - ref_pos[c];
        delta = box.wrap(delta);

        // Add the vector into the COM tally so far
        m_cluster_com[c] += delta;

        m_cluster_size[c]++;
    }

    // Now that we have totaled all of the cluster vectors, compute the COM
    // position by averaging and then shifting by ref_pos
    for (unsigned int c = 0; c < m_num_clusters; c++)
    {
        float s = float(m_cluster_size[c]);
        vec3<float> v = m_cluster_com[c] / s + ref_pos[c];
        m_cluster_com[c] = box.wrap(v);
    }

    // Now that we have determined the centers of mass for each cluster, tally
    // up the G tensor. This has to be done in a loop over the particles, again
    for (unsigned int i = 0; i < Np; i++)
    {
        unsigned int c = cluster_idx[i];
        vec3<float> pos = points[i];
        vec3<float> delta = box.wrap(pos - m_cluster_com[c]);

        // get the start pointer for our 3x3 matrix
        m_cluster_G(c, 0, 0) += delta.x * delta.x;
        m_cluster_G(c, 0, 1) += delta.x * delta.y;
        m_cluster_G(c, 0, 2) += delta.x * delta.z;
        m_cluster_G(c, 1, 0) += delta.y * delta.x;
        m_cluster_G(c, 1, 1) += delta.y * delta.y;
        m_cluster_G(c, 1, 2) += delta.y * delta.z;
        m_cluster_G(c, 2, 0) += delta.z * delta.x;
        m_cluster_G(c, 2, 1) += delta.z * delta.y;
        m_cluster_G(c, 2, 2) += delta.z * delta.z;
    }

    // Normalize by the cluster size.
    for (unsigned int c = 0; c < m_num_clusters; c++)
    {
        float s = float(m_cluster_size[c]);
        m_cluster_G(c, 0, 0) /= s;
        m_cluster_G(c, 0, 1) /= s;
        m_cluster_G(c, 0, 2) /= s;
        m_cluster_G(c, 1, 0) /= s;
        m_cluster_G(c, 1, 1) /= s;
        m_cluster_G(c, 1, 2) /= s;
        m_cluster_G(c, 2, 0) /= s;
        m_cluster_G(c, 2, 1) /= s;
        m_cluster_G(c, 2, 2) /= s;
    }
}

}; }; // end namespace freud::cluster
