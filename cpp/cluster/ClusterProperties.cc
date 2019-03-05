// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

#include "ClusterProperties.h"

using namespace std;

/*! \file ClusterProperties.cc
    \brief Routines for computing properties of point clusters.
*/

namespace freud { namespace cluster {

ClusterProperties::ClusterProperties()
    : m_num_clusters(0)
    {
    }

/*! \param box Box containing the particles
    \param points Positions of the particles making up the clusters
    \param cluster_idx Index of which cluster each point belongs to
    \param Np Number of particles (length of \a points and \a cluster_idx)

    computeClusterProperties loops over all points in the given array and determines the center of mass of the cluster
    as well as the G tensor. These can be accessed after the call to compute with getClusterCOM() and getClusterG().
*/
void ClusterProperties::computeProperties(const box::Box& box,
                                          const vec3<float> *points,
                                          const unsigned int *cluster_idx,
                                          unsigned int Np)
    {
    assert(points);
    assert(cluster_idx);
    assert(Np > 0);

    // determine the number of clusters
    const unsigned int *max_cluster_id = max_element(cluster_idx, cluster_idx+Np);
    m_num_clusters = *max_cluster_id+1;

    // allocate memory for the cluster properties and temporary arrays
    // initialize arrays to 0
    m_cluster_com = std::shared_ptr< vec3<float> >(
            new vec3<float>[m_num_clusters],
            std::default_delete< vec3<float>[]>());
    memset((void*)m_cluster_com.get(), 0, sizeof(vec3<float>)*m_num_clusters);

    m_cluster_G = std::shared_ptr<float>(
            new float[m_num_clusters*3*3],
            std::default_delete<float[]>());
    memset((void*)m_cluster_G.get(), 0, sizeof(float)*m_num_clusters*3*3);

    m_cluster_size = std::shared_ptr<unsigned int>(
            new unsigned int[m_num_clusters],
            std::default_delete<unsigned int[]>());
    memset((void*)m_cluster_size.get(), 0, sizeof(unsigned int)*m_num_clusters);

    // ref_particle is the first particle found in a cluster, it is used as a
    // reference to compute the COM in relation to, for handling of the
    // periodic boundary conditions
    vector< vec3<float> > ref_pos(m_num_clusters, vec3<float>(0.0f, 0.0f, 0.0f));
    // determine if we have seen this cluster before or not (used to initialize ref_pos)
    vector<bool> cluster_seen(m_num_clusters, false);

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
        m_cluster_com.get()[c] += delta;

        m_cluster_size.get()[c]++;
        }

    // Now that we have totaled all of the cluster vectors, compute the COM
    // position by averaging and then shifting by ref_pos
    for (unsigned int c = 0; c < m_num_clusters; c++)
        {
        float s = float(m_cluster_size.get()[c]);
        vec3<float> v = m_cluster_com.get()[c] / s + ref_pos[c];
        m_cluster_com.get()[c] = box.wrap(v);
        }

    // Now that we have determined the centers of mass for each cluster, tally
    // up the G tensor. This has to be done in a loop over the particles, again
    for (unsigned int i = 0; i < Np; i++)
        {
        unsigned int c = cluster_idx[i];
        vec3<float> pos = points[i];
        vec3<float> delta = box.wrap(pos - m_cluster_com.get()[c]);

        // get the start pointer for our 3x3 matrix
        float *G = m_cluster_G.get() + c*9;
        G[0*3+0] += delta.x * delta.x;
        G[0*3+1] += delta.x * delta.y;
        G[0*3+2] += delta.x * delta.z;
        G[1*3+0] += delta.y * delta.x;
        G[1*3+1] += delta.y * delta.y;
        G[1*3+2] += delta.y * delta.z;
        G[2*3+0] += delta.z * delta.x;
        G[2*3+1] += delta.z * delta.y;
        G[2*3+2] += delta.z * delta.z;
        }

    // now need to divide by the number of particles in each cluster
    for (unsigned int c = 0; c < m_num_clusters; c++)
        {
        float *G = m_cluster_G.get() + c*9;
        float s = float(m_cluster_size.get()[c]);
        G[0*3+0] /= s;
        G[0*3+1] /= s;
        G[0*3+2] /= s;
        G[1*3+0] /= s;
        G[1*3+1] /= s;
        G[1*3+2] /= s;
        G[2*3+0] /= s;
        G[2*3+1] /= s;
        G[2*3+2] /= s;
        }

    // done!
    }

}; }; // end namespace freud::cluster
