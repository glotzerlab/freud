// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef SOL_LIQ_H
#define SOL_LIQ_H

#include <algorithm>
#include <complex>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <stdint.h>
#include <vector>

#include "Box.h"
#include "VectorMath.h"
#include "Cluster.h"
#include "LinkCell.h"
#include "fsph/src/spherical_harmonics.hpp"

namespace freud { namespace order {

//! Computes dot products of Qlm between particles and uses these for clustering

class SolLiq
    {
    public:
        //! Constructor
        /*! Constructor for Solid-Liquid analysis class. After creation, call
         *  compute to calculate solid-like clusters. Use accessor functions
         *  to retrieve data.
         *  \param box A freud box for the trajectory.
         *  \param rmax Cutoff radius for cell list and clustering algorithm.
              Values near first minima of the rdf are recommended.
         *  \param Qthreshold Value of dot product threshold when evaluating
               \f$Q_{lm}^*(i) Q_{lm}(j)\f$ to determine if a neighbor pair is
               a solid-like bond. (For l=6, 0.7 generally good for FCC or BCC
               structures)
         *  \param Sthreshold Minimum required number of adjacent solid-link bonds
               for a particle to be considered solid-like for clustering. (For
               l=6, 6-8 generally good for FCC or BCC structures)
         *  \param l Choose spherical harmonic Ql. Must be positive and even.
        **/
        SolLiq(const box::Box& box, float rmax, float Qthreshold, unsigned int Sthreshold, unsigned int l);

        //! Get the simulation box
        const box::Box& getBox()
            {
            return m_box;
            }

        //! Reset the simulation box size
        void setBox(const box::Box newbox)
            {
            m_box = newbox;
            }


        //! Reset the simulation box size
        void setClusteringRadius(float rcut_cluster)
            {
            if (rcut_cluster < m_rmax)
                throw std::invalid_argument("SolLiq requires that rcut_cluster must be greater than rcut (for local env).");
                //May not be necessary if std::max(m_rmax, m_rmax_cluster) is used to rebuild cell list here, and in setBox.

            m_rmax_cluster = rcut_cluster;
            }

        //! Compute the Solid-Liquid Order Parameter
        void compute(const locality::NeighborList *nlist, const vec3<float> *points, unsigned int Np);

        //! Try to cluster requiring particles to have S_threshold number of
        //  shared neighbors to be clustered. This enforces stronger
        //  conditions on orientations.
        void computeSolLiqVariant(const locality::NeighborList *nlist, const vec3<float> *points, unsigned int Np);

        //! Compute Solid-Liquid order parameter without normalizing the dot
        //  product. This is for comparisons with literature.
        void computeSolLiqNoNorm(const locality::NeighborList *nlist, const vec3<float> *points, unsigned int Np);

        //! Calculates spherical harmonic Ylm for given theta, phi using fsph.
        void Ylm(const float theta, const float phi, std::vector<std::complex<float> > &Y);

        //! Returns largest cluster size! Please compute solliq first!
        unsigned int getLargestClusterSize();

        //! Returns a vector containing the size of all clusters.
        std::vector<unsigned int> getClusterSizes();

        //! Get a reference to the last computed Qlmi
        std::shared_ptr< std::complex<float> > getQlmi()
            {
            return m_Qlmi_array;
            }

        //! Get a reference to the last computed set of solid-like cluster
        //  indices for each particle
        std::shared_ptr<unsigned int > getClusters()
            {
            return m_cluster_idx;
            }

        //! Get a reference to the number of connections per particle
        std::shared_ptr<unsigned int> getNumberOfConnections()
            {
            return m_number_of_connections;
            }

        //! Get a reference to the Qldot_ij values
        std::vector<std::complex<float> > getQldot_ij()
            {
            return m_qldot_ij;
            }

        unsigned int getNP()
            {
            return m_Np;
            }

        unsigned int getNumClusters()
            {
            return m_num_clusters;
            }

    private:
        //Calculates Qlmi
        void computeClustersQ(const locality::NeighborList *nlist,
                              const vec3<float> *points,
                              unsigned int Np);
        //! Computes the number of solid-like neighbors based on the dot product thresholds
        void computeClustersQdot(const locality::NeighborList *nlist,
                                 const vec3<float> *points,
                              unsigned int Np);

        //!Clusters particles based on values of Q_l dot product and solid-like neighbor thresholds
        void computeClustersQS(const locality::NeighborList *nlist,
                               const vec3<float> *points,
                              unsigned int Np);

        //Compute list of solidlike neighbors
        void computeListOfSolidLikeNeighbors(const locality::NeighborList *nlist,
                                             const vec3<float> *points,
                                             unsigned int Np,
                                             std::vector<std::vector<unsigned int> > &SolidlikeNeighborlist);

        //Alternate clustering method requiring same shared neighbors
        void computeClustersSharedNeighbors(const locality::NeighborList *nlist,
                                            const vec3<float> *points,
                                            unsigned int Np,
                                            const std::vector<std::vector<unsigned int> > &SolidlikeNeighborlist);

        void computeClustersQdotNoNorm(const locality::NeighborList *nlist,
                                       const vec3<float> *points,
                              unsigned int Np);

        box::Box m_box;        //!< Simulation box where the particles belong
        float m_rmax;          //!< Maximum cutoff radius at which to determine local environment
        float m_rmax_cluster;  //!< Maximum radius at which to cluster solid-like particles;

        unsigned int m_Np;     //!< Last number of points computed
        std::shared_ptr< std::complex<float> > m_Qlmi_array;    //!< Stores Qlm for each particle i
        float m_Qthreshold;           //!< Dotproduct cutoff
        unsigned int m_Sthreshold;    //!< Solid-like num connections cutoff
        unsigned int m_l;             //!< Value of l for the spherical harmonic.

        // Pull cluster data into these
        unsigned int m_num_clusters;    //!< Number of clusters found in the last call to compute()
        std::shared_ptr<unsigned int> m_cluster_idx;    //!< Cluster index determined for each particle
        std::vector< std::complex<float> > m_qldot_ij;  //!< All of the Qlmi dot Qlmj's computed
        //! Number of connections for each particle with dot product above Qthreshold
        std::shared_ptr<unsigned int> m_number_of_connections;
        //! Number of neighbors for each particle (used for normalizing spherical harmonics)
        std::shared_ptr<unsigned int> m_number_of_neighbors;
        //! Stores number of shared neighbors for all ij pairs considered
        std::vector<unsigned int> m_number_of_shared_connections;
    };

}; }; // end namespace freud::order

#endif // SOL_LIQ_H
