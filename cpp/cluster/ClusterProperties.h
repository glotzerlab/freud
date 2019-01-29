// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef CLUSTER_PROPERTIES_H
#define CLUSTER_PROPERTIES_H

#include <memory>

#include "Box.h"

/*! \file ClusterProperties.h
    \brief Routines for computing properties of point clusters.
*/

namespace freud { namespace cluster {

//! Computes properties of clusters
/*! Given a set of points and \a cluster_idx (from Cluster, or some other
    source), ClusterProperties determines the following properties for each
    cluster:
     - Center of mass
     - Gyration radius tensor

    m_cluster_com stores the computed center of mass for each cluster
    (properly handling periodic boundary conditions, of course). It is an
    array of vec3<float>'s in C++. It is passed to Python from getClusterCOM
    as a num_clusters x 3 numpy array.

    m_cluster_G stores a 3x3 G tensor for each cluster. Index cluster \a c,
    element \a j, \a i with the following:
    m_cluster_G[c*9 + j*3 + i]. The tensor is symmetric, so the choice of i
    and j are irrelevant. This is passed back to Python as a
    num_clusters x 3 x 3 numpy array.
*/
class ClusterProperties
    {
    public:
        //! Constructor
        ClusterProperties();

        //! Compute properties of the point clusters
        void computeProperties(const box::Box& box,
                               const vec3<float> *points,
                               const unsigned int *cluster_idx,
                               unsigned int Np);

        //! Count the number of clusters found in the last call to computeProperties()
        unsigned int getNumClusters()
            {
            return m_num_clusters;
            }

        //! Get a reference to the last computed cluster_com
        std::shared_ptr< vec3<float> > getClusterCOM()
            {
            return m_cluster_com;
            }

        //! Get a reference to the last computed cluster_G
        std::shared_ptr<float> getClusterG()
            {
            return m_cluster_G;
            }

        //! Get a reference to the last computed cluster size
        std::shared_ptr<unsigned int> getClusterSize()
            {
            return m_cluster_size;
            }

    private:
        unsigned int m_num_clusters;                   //!< Number of clusters found in the last call to computeProperties()
        std::shared_ptr< vec3<float> > m_cluster_com;  //!< Center of mass computed for each cluster (length: m_num_clusters)
        std::shared_ptr<float> m_cluster_G;            //!< Gyration tensor computed for each cluster (m_num_clusters x 3 x 3 array)
        std::shared_ptr<unsigned int> m_cluster_size;  //!< Size per cluster
    };

}; }; // end namespace freud::cluster

#endif // CLUSTER_PROPERTIES_H
