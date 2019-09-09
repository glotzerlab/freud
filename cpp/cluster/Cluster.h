// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef CLUSTER_H
#define CLUSTER_H

#include <memory>
#include <set>
#include <stdint.h>
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
class Cluster
{
public:
    //! Constructor
    Cluster();

    //! Compute the point clusters
    void compute(const freud::locality::NeighborQuery* nq,
                 const freud::locality::NeighborList* nlist,
                 freud::locality::QueryArgs qargs,
                 const unsigned int* keys=NULL);

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

    //! Get a reference to the last computed cluster ids
    const util::ManagedArray<unsigned int> &getClusterIdx()
    {
        return m_cluster_idx;
    }

    //! Returns the last computed cluster keys
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
