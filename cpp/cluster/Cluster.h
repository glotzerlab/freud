// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef CLUSTER_H
#define CLUSTER_H

#include <vector>

#include "Box.h"
#include "ManagedArray.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "VectorMath.h"

/*! \file Cluster.h
    \brief Routines for clustering points.
*/

namespace freud { namespace cluster {
//! Finds clusters using a network of neighbors.
/*! Given a set of particles and their neighbors,
 *  freud.cluster.Cluster will determine all of the connected
 *  components of the network formed by those neighbor bonds. That is, two
 *  points are in the same cluster if and only if a path exists between them on
 *  the network of bonds. The class attribute cluster_idx holds an
 *  array of cluster indices for each particle. By the definition of a cluster,
 *  points that are not bonded to any other point end up in their own
 *  1-particle cluster.
 *
 *  Identifying micelles is one use-case for finding clusters. This operation
 *  is somewhat different, though. In a cluster of points, each and every point
 *  belongs to one and only one cluster. However, because a string of points
 *  belongs to a polymer, that single polymer may be present in more than one
 *  cluster. To handle this situation, an optional layer is presented on top of
 *  the cluster_idx array. Given a key value per particle (e.g. the polymer
 *  id), the compute function will process clusters with the key values in mind and
 *  provide a list of keys that are present in each cluster in the attribute
 *  cluster_keys, as a list of lists. If keys are not provided, every
 *  particle is assigned a key corresponding to its index, and cluster_keys
 *  contains the particle ids present in each cluster.
 *
 *  <b>2D:</b><br>
 *  Cluster properly handles 2D boxes. As with everything else in freud, 2D
 *  points must be passed in as 3 component vectors x, y, 0. Failing to set 0 in
 *  the third component will lead to undefined behavior.
 */
class Cluster
{
public:
    //! Constructor
    Cluster();

    //! Compute the point clusters.
    void compute(const freud::locality::NeighborQuery* nq,
                 const freud::locality::NeighborList* nlist,
                 freud::locality::QueryArgs qargs,
                 const unsigned int* keys=NULL);

    //! Count the number of clusters found in the last call to compute().
    unsigned int getNumClusters()
    {
        return m_num_clusters;
    }

    //! Return the number of particles in the current Compute.
    unsigned int getNumParticles()
    {
        return m_num_particles;
    }

    //! Get a reference to the last computed cluster ids.
    const util::ManagedArray<unsigned int> &getClusterIdx()
    {
        return m_cluster_idx;
    }

    //! Returns the last computed cluster keys.
    const std::vector<std::vector<unsigned int>> &getClusterKeys()
    {
        return m_cluster_keys;
    }

private:
    unsigned int m_num_particles; //!< Number of particles processed in the last call to compute()
    unsigned int m_num_clusters;  //!< Number of clusters found in the last call to compute()
    util::ManagedArray<unsigned int> m_cluster_idx; //!< Cluster index determined for each particle
    std::vector<std::vector<unsigned int>> m_cluster_keys; //!< List of keys in each cluster

    // Returns inverse permutation of cluster indices, sorted from largest to
    // smallest. Adapted from
    // https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
    static std::vector<size_t> sort_indexes_inverse(const std::vector<size_t> &counts,
            const std::vector<size_t> &min_ids);
};

}; }; // end namespace freud::cluster

#endif // CLUSTER_H
