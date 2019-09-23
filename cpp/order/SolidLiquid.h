// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef SOLID_LIQUID_H
#define SOLID_LIQUID_H

#include <algorithm>
#include <complex>
#include <memory>
#include <stdexcept>
#include <stdint.h>
#include <vector>

#include "Box.h"
#include "Cluster.h"
#include "ManagedArray.h"
#include "NeighborList.h"
#include "NeighborComputeFunctional.h"
#include "Steinhardt.h"
#include "ThreadStorage.h"
#include "VectorMath.h"

namespace freud { namespace order {

//! Computes dot products of Qlm between particles and uses these for clustering

class SolidLiquid
{
public:
    //! Constructor
    /*! Constructor for Solid-Liquid analysis class. After creation, call
     *  compute to calculate solid-like clusters. Use accessor functions
     *  to retrieve data.
     *  \param l Choose spherical harmonic Ql. Must be positive and even.
     *  \param Q_threshold Value of dot product threshold when evaluating
     *     \f$Q_{lm}^*(i) Q_{lm}(j)\f$ to determine if a neighbor pair is
     *     a solid-like bond. (For l=6, 0.7 generally good for FCC or BCC
     *     structures)
     *  \param S_threshold Minimum required number of adjacent solid-link bonds
     *     for a particle to be considered solid-like for clustering. (For
     *     l=6, 6-8 generally good for FCC or BCC structures)
     *  \param normalize_Q Whether to normalize the per-bond dot products of Qlm.
     *  \param common_neighbors Whether to use common neighbors for clustering.
     */
    SolidLiquid(unsigned int l, float Q_threshold, unsigned int S_threshold, bool normalize_Q=true, bool common_neighbors=false);

    unsigned int getL()
    {
        return m_l;
    }

    float getQThreshold()
    {
        return m_Q_threshold;
    }

    unsigned int getSThreshold()
    {
        return m_S_threshold;
    }

    bool getNormalizeQ()
    {
        return m_normalize_Q;
    }

    bool getCommonNeighbors()
    {
        return m_common_neighbors;
    }

    //! Compute the Solid-Liquid Order Parameter
    void compute(const freud::locality::NeighborList* nlist,
            const freud::locality::NeighborQuery* points, freud::locality::QueryArgs qargs);

    //! Returns the Ql_i_dot_j values for each bond
    const util::ManagedArray<std::complex<float>> &getQlij()
    {
        return m_Ql_i_dot_j;
    }

    //! Returns largest cluster size.
    unsigned int getLargestClusterSize()
    {
        return m_cluster.getClusterKeys()[0].size();
    }

    //! Returns a vector containing the size of all clusters.
    std::vector<unsigned int> getClusterSizes()
    {
        std::vector<unsigned int> sizes;
        auto keys = m_cluster.getClusterKeys();
        for (auto cluster = keys.begin(); cluster != keys.end(); cluster++)
        {
            sizes.push_back(cluster->size());
        }
        return sizes;
    }

    //! Get a reference to the last computed set of solid-like cluster
    //  indices for each particle
    const util::ManagedArray<unsigned int> &getClusterIdx()
    {
        return m_cluster.getClusterIdx();
    }

    //! Get a reference to the number of connections per particle
    const util::ManagedArray<unsigned int> &getNumberOfConnections()
    {
        return m_number_of_connections;
    }

    unsigned int getNumClusters()
    {
        return m_cluster.getNumClusters();
    }

private:
    unsigned int m_l;                       //!< Value of l for the spherical harmonic.
    unsigned int m_num_ms;                  //!< The number of magnetic quantum numbers (2*m_l+1).
    float m_Q_threshold;                    //!< Dot product cutoff
    unsigned int m_S_threshold;             //!< Solid-like num connections cutoff
    bool m_normalize_Q;                     //!< Whether to normalize the Qlmi dot products.
    bool m_common_neighbors;                //!< Whether to threshold on common neighbors.

    freud::order::Steinhardt m_steinhardt;  //!< Steinhardt class used to compute Qlm
    freud::cluster::Cluster m_cluster;      //!< Cluster class used to cluster solid-like bonds

    util::ManagedArray<std::complex<float>> m_Ql_i_dot_j; //!< All of the Qlmi dot Qlmj's computed
    //! Number of connections for each particle with dot product above Q_threshold
    util::ManagedArray<unsigned int> m_number_of_connections;
    util::ThreadStorage<unsigned int> m_number_of_connections_local;

    //! Number of neighbors for each particle (used for normalizing spherical harmonics)
    util::ManagedArray<unsigned int> m_number_of_neighbors;
    //! Stores number of shared neighbors for all ij pairs considered
    std::vector<unsigned int> m_number_of_shared_connections;
};

}; }; // end namespace freud::order

#endif // SOLID_LIQUID_H
