// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef CLUSTER_H
#define CLUSTER_H

#include <memory>
#include <set>
#include <stdint.h>
#include <vector>

#include "Box.h"
#include "VectorMath.h"
#include "LinkCell.h"

/*! \file Cluster.h
    \brief Routines for clustering points.
*/

namespace freud { namespace cluster {

//! A disjoint set
/*! Implements efficient find and merge for disjoint sets
    Source of algorithms: Brassard and Bratley, _Fundamentals of Algorithmics_
*/
class DisjointSet
    {
    private:
        std::vector<uint32_t> s;            //!< The disjoint set data
        std::vector<unsigned int> rank;     //!< The rank of each tree in the set
    public:
        //! Constructor
        DisjointSet(uint32_t n = 0);
        //! Merge two sets
        void merge(const uint32_t a, const uint32_t b);
        //! Find the set with a given element
        uint32_t find(const uint32_t c);
    };

//! Find clusters in a set of points
/*! Given a set of coordinates and a cutoff, Cluster will determine all of the
    clusters of points that are made up of points that are closer than the
    cutoff. Clusters are labeled from 0 to the number of clusters-1 and an index
    array is returned where \c cluster_idx[i] is the cluster index in which
    particle \c i is found. By the definition of a cluster, points that are not
    within the cutoff of another point end up in their own 1-particle cluster.
    Identifying micelles is one primary use-case for finding clusters. This
    operation is somewhat different, though. In a cluster of points, each and
    every point belongs to one and only one cluster. However, because a string
    of points belongs to a polymer, that single polymer may be present in more
    than one cluster. To handle this situation, an optional layer is presented
    on top of the \c cluster_idx array. Given a key value per particle (i.e. the
    polymer id), the computeClusterMembership function will process cluster_idx
    with the key values in mind and provide a list of keys that are present in
    each cluster.

    <b>2D:</b><br>
    Cluster properly handles 2D boxes. As with everything else in freud, 2D
    points must be passed in as 3 component vectors x,y,0. Failing to set 0 in
    the third component will lead to undefined behavior.
*/
class Cluster
    {
    public:
        //! Constructor
        Cluster(float rcut);

        //! Compute the point clusters
        void computeClusters(const box::Box& box,
                             const freud::locality::NeighborList *nlist,
                             const vec3<float> *points,
                             unsigned int Np);

        //! Compute clusters with key membership
        void computeClusterMembership(const unsigned int *keys);

        //! Count the number of clusters found in the last call to compute()
        unsigned int getNumClusters()
            {
            return m_num_clusters;
            }

        //! Return the number of particles in the current Compute
        unsigned int getNumParticles()
            {
            return m_num_particles;
            }

        //! Get a reference to the last computed cluster_idx
        std::shared_ptr<unsigned int> getClusterIdx()
            {
            return m_cluster_idx;
            }

        //! Returns the cluster keys last determined by computeClusterKeys
        const std::vector< std::vector<unsigned int> >& getClusterKeys()
            {
            return m_cluster_keys;
            }

    private:
        float m_rcut;                    //!< Maximum r at which points will be counted in the same cluster
        unsigned int m_num_particles;    //!< Number of particles processed in the last call to compute()
        unsigned int m_num_clusters;     //!< Number of clusters found in the last call to compute()
        std::shared_ptr<unsigned int> m_cluster_idx;   //!< Cluster index determined for each particle
        std::vector< std::vector<unsigned int> > m_cluster_keys;   //!< List of keys in each cluster
    };

}; }; // end namespace freud::cluster

#endif // CLUSTER_H
