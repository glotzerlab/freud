// Copyright (c) 2010-2023 The Regents of the University of Michigan
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
    getClusterInertiaMoments().
*/

void ClusterProperties::compute(const freud::locality::NeighborQuery* nq, const unsigned int* cluster_idx,
                                const float* masses)
{
    // determine the number of clusters
    const unsigned int* max_cluster_id = std::max_element(cluster_idx, cluster_idx + nq->getNPoints());
    const unsigned int num_clusters = *max_cluster_id + 1;

    // allocate memory for the cluster properties and temporary arrays
    // initialize arrays to 0
    m_cluster_centers.prepare(num_clusters);
    m_cluster_centers_of_mass.prepare(num_clusters);
    m_cluster_moments_of_inertia.prepare({num_clusters, 3, 3});
    m_cluster_gyrations.prepare({num_clusters, 3, 3});
    m_cluster_sizes.prepare(num_clusters);
    m_cluster_masses.prepare(num_clusters);

    // Create a vector to store cluster points, used to compute center of mass
    std::vector<std::vector<vec3<float>>> cluster_points(num_clusters, std::vector<vec3<float>>());

    // Start by determining the center of mass of each cluster. Since we are
    // given an array of points, the easiest way to do this is to loop over
    // all points and add the appropriate information to m_cluster_centers as
    // we go.
    for (unsigned int i = 0; i < nq->getNPoints(); i++)
    {
        const unsigned int c = cluster_idx[i];
        cluster_points[c].push_back((*nq)[i]);
        m_cluster_sizes[c]++;
        float mass = (masses != nullptr) ? masses[i] : float(1.0);
        m_cluster_masses[c] += mass;
    }

    // Now that we have located all of the cluster vectors, compute the centers
    const float* cluster_point_masses = masses;
    for (unsigned int c = 0; c < num_clusters; c++)
    {
        m_cluster_centers[c] = nq->getBox().centerOfMass(cluster_points[c].data(), m_cluster_sizes[c]);
        m_cluster_centers_of_mass[c]
            = nq->getBox().centerOfMass(cluster_points[c].data(), m_cluster_sizes[c], cluster_point_masses);
        if (masses == nullptr)
        {
            cluster_point_masses = nullptr;
        }
        else
        {
            cluster_point_masses = cluster_point_masses + m_cluster_sizes[c];
        }
    }

    // Now that we have determined the centers of mass for each cluster, tally
    // up the moment of inertia tensor. This has to be done in a loop over the points.
    for (unsigned int i = 0; i < nq->getNPoints(); i++)
    {
        float mass = (masses != nullptr) ? masses[i] : float(1.0);
        unsigned int c = cluster_idx[i];
        vec3<float> pos = (*nq)[i];
        vec3<float> mass_delta = nq->getBox().wrap(pos - m_cluster_centers_of_mass[c]);
        vec3<float> delta = nq->getBox().wrap(pos - m_cluster_centers[c]);

        // get the start pointer for our 3x3 matrix
        m_cluster_moments_of_inertia(c, 0, 0)
            += (std::pow(mass_delta.y, 2) + std::pow(mass_delta.z, 2)) * mass;
        m_cluster_moments_of_inertia(c, 0, 1) -= mass_delta.x * mass_delta.y * mass;
        m_cluster_moments_of_inertia(c, 0, 2) -= mass_delta.x * mass_delta.z * mass;
        m_cluster_moments_of_inertia(c, 1, 0) -= mass_delta.y * mass_delta.x * mass;
        m_cluster_moments_of_inertia(c, 1, 1)
            += (std::pow(mass_delta.x, 2) + std::pow(mass_delta.z, 2)) * mass;
        m_cluster_moments_of_inertia(c, 1, 2) -= mass_delta.y * mass_delta.z * mass;
        m_cluster_moments_of_inertia(c, 2, 0) -= mass_delta.z * mass_delta.x * mass;
        m_cluster_moments_of_inertia(c, 2, 1) -= mass_delta.z * mass_delta.y * mass;
        m_cluster_moments_of_inertia(c, 2, 2)
            += (std::pow(mass_delta.x, 2) + std::pow(mass_delta.y, 2)) * mass;

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

    // Normalize by the cluster sizes.
    for (unsigned int c = 0; c < num_clusters; c++)
    {
        auto s = static_cast<float>(m_cluster_sizes[c]);
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
