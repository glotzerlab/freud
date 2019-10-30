// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <vector>

#include "ClusterProperties.h"
#include "NeighborComputeFunctional.h"

/*! \file ClusterProperties.cc
    \brief Routines for computing properties of point clusters.
*/

namespace freud { namespace cluster {

/*! \param nq NeighborQuery containing the points making up the clusters
    \param cluster_idx Index of which cluster each point belongs to

    compute loops over all points in the given array and determines the center
    of mass of the cluster as well as the gyration tensor. These can be
    accessed after the call to compute with getClusterCenters() and
    getClusterGyrations().
*/

void ClusterProperties::compute(const freud::locality::NeighborQuery* nq, const unsigned int* cluster_idx)
{
    // determine the number of clusters
    const unsigned int* max_cluster_id = std::max_element(cluster_idx, cluster_idx + nq->getNPoints());
    const unsigned int num_clusters = *max_cluster_id + 1;

    // allocate memory for the cluster properties and temporary arrays
    // initialize arrays to 0
    m_cluster_centers.prepare(num_clusters);
    m_cluster_gyrations.prepare({num_clusters, 3, 3});
    m_cluster_sizes.prepare(num_clusters);

    // ref_pos is the first point found in a cluster, it is used as a reference
    // to compute the center in relation to, for handling of the periodic
    // boundary conditions
    std::vector<vec3<float>> ref_pos(num_clusters);
    // determine if we have seen this cluster before or not (used to initialize ref_pos)
    std::vector<bool> cluster_seen(num_clusters, false);

    // Start by determining the center of mass of each cluster. Since we are
    // given an array of points, the easiest way to do this is to loop over
    // all points and add the appropriate information to m_cluster_centers as
    // we go.
    for (unsigned int i = 0; i < nq->getNPoints(); i++)
    {
        const unsigned int c = cluster_idx[i];

        // The first time we see the cluster, mark a reference position
        if (!cluster_seen[c])
        {
            ref_pos[c] = (*nq)[i];
            cluster_seen[c] = true;
        }

        // To compute the center in periodic boundary conditions, compute all
        // reference vectors as wrapped vectors relative to ref_pos. When we
        // are done, we can add the computed center to ref_pos to get the
        // center in the space frame.
        const vec3<float> delta(bondVector(locality::NeighborBond(c, i), nq, ref_pos.data()));

        // Add the vector into the center tally so far
        m_cluster_centers[c] += delta;

        m_cluster_sizes[c]++;
    }

    // Now that we have totaled all of the cluster vectors, compute the center
    // position by averaging and then shifting by ref_pos
    for (unsigned int c = 0; c < num_clusters; c++)
    {
        float s = float(m_cluster_sizes[c]);
        vec3<float> v = m_cluster_centers[c] / s + ref_pos[c];
        m_cluster_centers[c] = nq->getBox().wrap(v);
    }

    // Now that we have determined the centers of mass for each cluster, tally
    // up the gyration tensor. This has to be done in a loop over the points.
    for (unsigned int i = 0; i < nq->getNPoints(); i++)
    {
        unsigned int c = cluster_idx[i];
        vec3<float> pos = (*nq)[i];
        vec3<float> delta = nq->getBox().wrap(pos - m_cluster_centers[c]);

        // get the start pointer for our 3x3 matrix
        m_cluster_gyrations(c, 0, 0) += delta.x * delta.x;
        m_cluster_gyrations(c, 0, 1) += delta.x * delta.y;
        m_cluster_gyrations(c, 0, 2) += delta.x * delta.z;
        m_cluster_gyrations(c, 1, 0) += delta.y * delta.x;
        m_cluster_gyrations(c, 1, 1) += delta.y * delta.y;
        m_cluster_gyrations(c, 1, 2) += delta.y * delta.z;
        m_cluster_gyrations(c, 2, 0) += delta.z * delta.x;
        m_cluster_gyrations(c, 2, 1) += delta.z * delta.y;
        m_cluster_gyrations(c, 2, 2) += delta.z * delta.z;
    }

    // Normalize by the cluster size.
    for (unsigned int c = 0; c < num_clusters; c++)
    {
        float s = float(m_cluster_sizes[c]);
        m_cluster_gyrations(c, 0, 0) /= s;
        m_cluster_gyrations(c, 0, 1) /= s;
        m_cluster_gyrations(c, 0, 2) /= s;
        m_cluster_gyrations(c, 1, 0) /= s;
        m_cluster_gyrations(c, 1, 1) /= s;
        m_cluster_gyrations(c, 1, 2) /= s;
        m_cluster_gyrations(c, 2, 0) /= s;
        m_cluster_gyrations(c, 2, 1) /= s;
        m_cluster_gyrations(c, 2, 2) /= s;
    }
}

}; }; // end namespace freud::cluster
