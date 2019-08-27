// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef SOLID_LIQUID_H
#define SOLID_LIQUID_H

#include <algorithm>
#include <complex>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <stdint.h>
#include <vector>

#include "Box.h"
#include "Cluster.h"
#include "NeighborList.h"
#include "Steinhardt.h"
#include "ThreadStorage.h"
#include "VectorMath.h"
#include "fsph/src/spherical_harmonics.hpp"

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

    //! Try to cluster requiring particles to have S_threshold number of
    //  shared neighbors to be clustered. This enforces stronger
    //  conditions on orientations.
    void computeSolidLiquidVariant(const locality::NeighborList* nlist, const vec3<float>* points,
                              unsigned int Np);

    //! Compute Solid-Liquid order parameter without normalizing the dot
    //  product. This is for comparisons with literature.
    void computeSolidLiquidNoNorm(const locality::NeighborList* nlist, const vec3<float>* points, unsigned int Np);

    //! Returns largest cluster size.
    unsigned int getLargestClusterSize();

    //! Returns a vector containing the size of all clusters.
    std::vector<unsigned int> getClusterSizes();

    //! Get a reference to the last computed set of solid-like cluster
    //  indices for each particle
    std::shared_ptr<unsigned int> getClusters()
    {
        return m_cluster_idx;
    }

    //! Get a reference to the number of connections per particle
    std::shared_ptr<unsigned int> getNumberOfConnections()
    {
        return m_number_of_connections;
    }

    unsigned int getNumClusters()
    {
        return m_num_clusters;
    }

private:
<<<<<<< HEAD
    // Calculates Qlmi
    void computeClustersQ(const locality::NeighborList* nlist, const vec3<float>* points, unsigned int Np);
    //! Computes the number of solid-like neighbors based on the dot product thresholds
    void computeClustersQdot(const locality::NeighborList* nlist, const vec3<float>* points, unsigned int Np);

    //! Clusters particles based on values of Q_l dot product and solid-like neighbor thresholds
    void computeClustersQS(const locality::NeighborList* nlist, const vec3<float>* points, unsigned int Np);

    // Compute list of solidlike neighbors
    void computeListOfSolidLikeNeighbors(const locality::NeighborList* nlist, const vec3<float>* points,
                                         unsigned int Np,
                                         std::vector<std::vector<unsigned int>>& SolidlikeNeighborlist);

    // Alternate clustering method requiring same shared neighbors
    void computeClustersSharedNeighbors(const locality::NeighborList* nlist, const vec3<float>* points,
                                        unsigned int Np,
                                        const std::vector<std::vector<unsigned int>>& SolidlikeNeighborlist);

    void computeClustersQdotNoNorm(const locality::NeighborList* nlist, const vec3<float>* points,
                                   unsigned int Np);

    void reduceNumberOfConnections(unsigned int Np);

    box::Box m_box;       //!< Simulation box where the particles belong
    float m_r_max;         //!< Maximum cutoff radius at which to determine local environment
    float m_r_max_cluster; //!< Maximum radius at which to cluster solid-like particles;

    unsigned int m_Np;                                 //!< Last number of points computed
    std::shared_ptr<std::complex<float>> m_Qlmi_array; //!< Stores Qlm for each particle i
    float m_Qthreshold;                                //!< Dotproduct cutoff
    unsigned int m_Sthreshold;                         //!< Solid-like num connections cutoff
=======
>>>>>>> dfd2e891... Intermediate work on API and refactoring Steinhardt to be a useful member object.
    unsigned int m_l;                                  //!< Value of l for the spherical harmonic.
    float m_Q_threshold;                               //!< Dot product cutoff
    unsigned int m_S_threshold;                        //!< Solid-like num connections cutoff
    bool m_normalize_Q;                                //!< Whether to normalize the Qlmi dot products.
    bool m_common_neighbors;                           //!< Whether to threshold on common neighbors.

    freud::order::Steinhardt m_steinhardt;                     //!< Steinhardt class used to compute Qlm
    freud::cluster::Cluster m_cluster;                           //!< Cluster class used to cluster solid-like bonds

    // Pull cluster data into these
    unsigned int m_num_clusters;                 //!< Number of clusters found in the last call to compute()
    std::shared_ptr<unsigned int> m_cluster_idx; //!< Cluster index determined for each particle
    std::vector<std::complex<float>> m_qldot_ij; //!< All of the Qlmi dot Qlmj's computed
    //! Number of connections for each particle with dot product above Q_threshold
    std::shared_ptr<unsigned int> m_number_of_connections;
    util::ThreadStorage<unsigned int> m_number_of_connections_local;

    //! Number of neighbors for each particle (used for normalizing spherical harmonics)
    std::shared_ptr<unsigned int> m_number_of_neighbors;
    //! Stores number of shared neighbors for all ij pairs considered
    std::vector<unsigned int> m_number_of_shared_connections;
};

}; }; // end namespace freud::order

#endif // SOLID_LIQUID_H
