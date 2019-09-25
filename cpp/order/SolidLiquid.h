// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef SOLID_LIQUID_H
#define SOLID_LIQUID_H

#include <complex>
#include <vector>

#include "Cluster.h"
#include "ManagedArray.h"
#include "NeighborList.h"
#include "Steinhardt.h"
#include "ThreadStorage.h"
#include "utils.h"

namespace freud { namespace order {

//! Identifies solid-like clusters using dot products of Q_{lm}.
/*! The solid-liquid order parameter (Frenkel 1995) uses a Steinhardt-like
 *  approach to identify solid-like particles. First, a bond parameter
 *  Q_l(i, j) is computed for each neighbor bond.
 *
 *  If normalize_Q is true (default), the bond parameter is given by
 *  Q_l(i, j) = \frac{\sum_{m=-l}^{l} \text{Re}~Q_{lm}(i) Q_{lm}^*(j)}
 *  {\sqrt{\sum_{m=-l}^{l} \lvert Q_{lm}(i) \rvert^2}
 *  \sqrt{\sum_{m=-l}^{l} \lvert Q_{lm}(j) \rvert^2}}
 *
 *  If normalize_Q is false, then the denominator of the above
 *  expression is left out.
 *
 *  Next, the bonds are filtered to keep only "solid-like" bonds with
 *  Q_l(i, j) above a cutoff value Q_{threshold}.
 *
 *  If a particle has more than S_{threshold} solid-like bonds, then
 *  the particle is considered solid-like. Finally, solid-like particles are
 *  clustered.
 *
 *  References:
 *  ten Wolde, P. R., Ruiz-Montero, M. J., & Frenkel, D. (1995).
 *  Numerical Evidence for bcc Ordering at the Surface of a Critical fcc Nucleus.
 *  Phys. Rev. Lett., 75 (2714). https://doi.org/10.1103/PhysRevLett.75.2714
 *
 *  Filion, L., Hermes, M., Ni, R., & Dijkstra, M. (2010).
 *  Crystal nucleation of hard spheres using molecular dynamics, umbrella sampling,
 *  and forward flux sampling: A comparison of simulation techniques.
 *  J. Chem. Phys. 133 (244115). https://doi.org/10.1063/1.3506838
 */

class SolidLiquid
{
public:
    //! Constructor
    /*! Constructor for Solid-Liquid analysis class.
     *  \param l Spherical harmonic number l.
     *  \param Q_threshold Value of dot product threshold when evaluating
     *     \f$Q_{lm}(i) Q_{lm}^*(j)\f$ to determine if a neighbor pair is
     *     a solid-like bond. (For l=6, 0.7 is generally good for FCC or BCC
     *     structures)
     *  \param S_threshold Minimum required number of adjacent solid-like bonds
     *     for a particle to be considered solid-like for clustering. (For
     *     l=6, 6-8 is generally good for FCC or BCC structures)
     *  \param normalize_Q Whether to normalize the per-bond dot products of Qlm.
     */
    SolidLiquid(unsigned int l, float Q_threshold, unsigned int S_threshold, bool normalize_Q=true);

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

    //! Compute the Solid-Liquid Order Parameter
    void compute(const freud::locality::NeighborList* nlist,
            const freud::locality::NeighborQuery* points, freud::locality::QueryArgs qargs);

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

    freud::order::Steinhardt m_steinhardt;  //!< Steinhardt class used to compute Qlm
    freud::cluster::Cluster m_cluster;      //!< Cluster class used to cluster solid-like bonds

    util::ManagedArray<float> m_Ql_ij; //!< All of the Qlmi dot Qlmj's computed
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
