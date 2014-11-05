#ifndef _SOL_LIQ_NEAR_H__
#define _SOL_LIQ_NEAR_H__

#include <boost/python.hpp>
#include <boost/shared_array.hpp>
//#include <boost/math/special_functions/spherical_harmonic.hpp>

#include "HOOMDMath.h"
#define swap freud_swap
#include "VectorMath.h"
#undef swap

#include <vector>
#include <set>

#include "Cluster.h"
#include "NearestNeighbors.h"
#include "num_util.h"

#include "trajectory.h"
#include <stdexcept>
#include <complex>
#include <map>
#include <algorithm>



namespace freud { namespace sphericalharmonicorderparameters {

//! Computes dot products of qlm between particles and uses these for clustering
/*!
*/

class SolLiqNear
    {
    public:
        //! Constructor
        /**Constructor for Solid-Liquid analysis class.  After creation, call compute to calculate solid-like clusters.  Use accessor functions to retrieve data.       
        @param box A freud box for the trajectory.
        @param rmax Cutoff radius for cell list and clustering algorithm.  Values near first minima of the rdf are recommended.
        @param Qthreshold Value of dot product threshold when evaluating \f$Q_{lm}^*(i) Q_{lm}(j)\f$ to determine if a neighbor pair is a solid-like bond. (For l=6, 0.7 generally good for FCC or BCC structures)
        @param Sthreshold Minimum required number of adjacent solid-link bonds for a particle to be considered solid-like for clustering. (For l=6, 6-8 generally good for FCC or BCC structures)
        @param l Choose spherical harmonic Ql.  Must be positive and even.
        **/
        SolLiqNear(const trajectory::Box& box, float rmax, float Qthreshold, unsigned int Sthreshold, unsigned int l, unsigned int kn);

        //! Get the simulation box
        const trajectory::Box& getBox()
            {
            return m_box;
            }

        //!  Reset the simulation box size
        void setBox(const trajectory::Box newbox)
            {
            m_box = newbox;  //Set
            locality::NearestNeighbors newNeighbors(m_box, std::max(m_rmax, m_rmax_cluster), m_k );  //Rebuild cell list
            m_nn = newNeighbors;
            } 


        //! Reset the simulation box size
        void setClusteringRadius(float rcut_cluster)
            {
            if (rcut_cluster < m_rmax)
                throw std::invalid_argument("rcut_cluster must be greater than rcut (for local env)");
                //May not be necessary if std::max(m_rmax, m_rmax_cluster) is used to rebuild cell list here, and in setBox.

            m_rmax_cluster = rcut_cluster;  //Set
            locality::NearestNeighbors newNeighbor(m_box, std::max(m_rmax, m_rmax_cluster), m_k );  //Rebuild cell list.
            m_nn = newNeighbor;
            }

        //! Compute the Solid-Liquid Order Parameter
        // void compute(const float3 *points, unsigned int Np);
        void compute(const vec3<float> *points, unsigned int Np);

        //! Try to cluster requiring particles to have S_threshold number of shared neighbors to be clustered.  This enforces stronger conditions on orientations.
        // void computeSolLiqVariant(const float3 *points, unsigned int Np);
        void computeSolLiqVariant(const vec3<float> *points, unsigned int Np);

        //! Compute Solid-Liquid order parameter without normalizing the dot product.  This is for comparisons with literature.
        // void computeSolLiqNoNorm(const float3 *points, unsigned int Np);
        void computeSolLiqNoNorm(const vec3<float> *points, unsigned int Np);

        //! Calculates spherical harmonic Y6m for given theta, phi using boost.
        void Ylm(const float theta, const float phi, std::vector<std::complex<float> > &Y);

        //! Calculates spherical harmonic Y6m for given theta, phi.
        void Y6m(const float theta, const float phi, std::vector<std::complex<float> > &Y);
        //! Calculates spherical harmonic Y4m for given theta, phi.
        void Y4m(const float theta, const float phi, std::vector<std::complex<float> > &Y);


        //! Returns largest cluster size! Please compute solliq first!
        unsigned int getLargestClusterSize();


        /*
        //!Python wrapper to obtain the largest cluster size from the last call to compute
        boost::python::numeric::array getLargestClusterSizePy()
            {
            unsigned int largestclustersize = getLargestClusterSize();
            unsigned int *arr = &largestclustersize;
            return num_util::makeNum(arr, 1);
            }

        */

        //! Returns a vector containing the size of all clusters.
        std::vector<unsigned int> getClusterSizes();

        //!Python wrapper to obtain list of all clusters sizes.
        boost::python::numeric::array getClusterSizesPy()
            {
            std::vector<unsigned int> clustersizes = getClusterSizes();
            //if(clustersizes.empty())
            //    {
            //    throw length_error("There are no clusters!");
            //    }
            unsigned int *arr = &clustersizes[0];
            return num_util::makeNum(arr, clustersizes.size());
            }

        //! Get a reference to the last computed Qlmi
        boost::shared_array< std::complex<float> > getQlmi()
            {
            return m_Qlmi_array;
            }
        //! Python wrapper for Qlmi() (returns a copy)
        boost::python::numeric::array getQlmiPy()
            {
            std::complex<float> *arr = m_Qlmi_array.get();
            return num_util::makeNum(arr, (2*m_l+1)*m_Np);
            }

        //! Python wrapper for compute
        void computePy(boost::python::numeric::array points);

        //! Python wrapper for variant solliq
        void computeSolLiqVariantPy(boost::python::numeric::array points);
        //! Python wrapper for variant
        void computeSolLiqNoNormPy(boost::python::numeric::array points);

        //! Expose to python a copy of the nonorm with scalar3 input.
        void computeNoNormVectorInputPy(boost::python::api::object &pyobj);

        //! Get a reference to the last computed set of solid-like cluster indices for each particle
        boost::shared_array<unsigned int > getClusters()
            {
            return m_cluster_idx;
            }
        //! Python wrapper for retrieving the last computed solid-like cluster indices for each particle (returns a copy)
        boost::python::numeric::array getClustersPy()
            {
            unsigned int *arr = m_cluster_idx.get();
            return num_util::makeNum(arr, m_Np);
            }

        //! Get a reference to the number of connections per particle
        boost::shared_array<unsigned int> getNumberOfConnections()
            {
            return m_number_of_connections;
            }
        //! Python wrapper for retrieving number of connections per particle
        boost::python::numeric::array getNumberOfConnectionsPy()
            {
            unsigned int *arr = m_number_of_connections.get();
            return num_util::makeNum(arr, m_Np);
            }

        //! Get a reference to the qldot_ij values
        std::vector<std::complex<float> > getQldot_ij()
            {
            return m_qldot_ij;
            }
        //! Python wrapper for retrieving number of connections per particle
        boost::python::numeric::array getQldot_ijPy()
            {
            std::complex<float> *arr = &m_qldot_ij.at(0);
            return num_util::makeNum(arr, m_qldot_ij.size());
            }

        //! Get a reference to the num shared solid-like neighbors from alternate compute method
        boost::python::numeric::array getNumberOfSharedConnectionsPy()
            {
            unsigned int *arr = &m_number_of_shared_connections.at(0);
            return num_util::makeNum(arr, m_number_of_shared_connections.size());
            }

        //! Python wrapper for using the Y4m calculation.  Returns array containing Yl for m=-l to l.
        boost::python::numeric::array calcYlmPy(float theta, float phi)
            {
            std::vector<std::complex<float> > Y;
            unsigned int length = 2*m_l+1; // 2*l+1
            Ylm(theta, phi,Y);
            boost::shared_array<std::complex<float> > arrY;
            arrY = boost::shared_array<std::complex<float> >(new std::complex<float> [Y.size()]);
            //Copy Y6m into a new vector!
            for(unsigned int i = 0; i < length; i++)
                {
                    arrY[i]=Y[i];
                }
            return num_util::makeNum(arrY.get(), length);
            }

        //! Python wrapper for using the Y6m calculation.  Returns array containing Y6 for m=-6 to 6.
        boost::python::numeric::array calcY6mPy(float theta, float phi)
            {
            std::vector<std::complex<float> > Y;
            unsigned int length = 2*6+1; // 2*l+1
            Y6m(theta, phi,Y);
            boost::shared_array<std::complex<float> > arrY;
            arrY = boost::shared_array<std::complex<float> >(new std::complex<float> [Y.size()]);
            //Copy Y6m into a new vector!
            for(unsigned int i = 0; i < length; i++)
                {
                    arrY[i]=Y[i];
                }
            return num_util::makeNum(arrY.get(), length);
            }

        //! Python wrapper for using the Y4m calculation.  Returns array containing Y4 for m=-4 to 4.
        boost::python::numeric::array calcY4mPy(float theta, float phi)
            {
            std::vector<std::complex<float> > Y;
            unsigned int length = 2*4+1; // 2*l+1
            Y4m(theta, phi,Y);
            boost::shared_array<std::complex<float> > arrY;
            arrY = boost::shared_array<std::complex<float> >(new std::complex<float> [Y.size()]);
            //Copy Y6m into a new vector!
            for(unsigned int i = 0; i < length; i++)
                {
                    arrY[i]=Y[i];
                }
            return num_util::makeNum(arrY.get(), length);
            }




    private:
        //Calculates Qlmi
        // void computeClustersQ(const float3 *points,
        //                       unsigned int Np);
        void computeClustersQ(const vec3<float> *points,
                              unsigned int Np);
        //! Computes the number of solid-like neighbors based on the dot product thresholds
        // void computeClustersQdot(const float3 *points,
        //                       unsigned int Np);
        void computeClustersQdot(const vec3<float> *points,
                              unsigned int Np);

        //!Clusters particles based on values of Q_l dot product and solid-like neighbor thresholds
        // void computeClustersQS(const float3 *points,
        //                       unsigned int Np);
        void computeClustersQS(const vec3<float> *points,
                              unsigned int Np);

        //Compute list of solidlike neighbors
        // void computeListOfSolidLikeNeighbors(const float3 *points,
        //                       unsigned int Np, std::vector< std::vector<unsigned int> > &SolidlikeNeighborlist);
        void computeListOfSolidLikeNeighbors(const vec3<float> *points,
                              unsigned int Np, std::vector< std::vector<unsigned int> > &SolidlikeNeighborlist);

        //Alternate clustering method requiring same shared neighbors
        // void computeClustersSharedNeighbors(const float3 *points,
        //                       unsigned int Np, const std::vector< std::vector<unsigned int> > &SolidlikeNeighborlist);
        void computeClustersSharedNeighbors(const vec3<float> *points,
                              unsigned int Np, const std::vector< std::vector<unsigned int> > &SolidlikeNeighborlist);

        // void computeClustersQdotNoNorm(const float3 *points,
        //                       unsigned int Np);
        void computeClustersQdotNoNorm(const vec3<float> *points,
                              unsigned int Np);

        trajectory::Box m_box;      //!< Simulation box the particles belong in
        float m_rmax;               //!< Maximum cutoff radius at which to determine local environment
        float m_rmax_cluster;       //!< Maximum radius at which to cluster solid-like particles;
        float m_k;                  //!< Number of neighbors
        locality::NearestNeighbors m_nn;    //!< NearestNeighbors to bin particles for the computation of local environments

        unsigned int m_Np;          //!< Last number of points computed
        boost::shared_array< std::complex<float> > m_Qlmi_array; //!< Stores Qlm for each particle i
        //boost::shared_array<float> m_Qli_array;  //!< Stores Ql rotationally invariant local order for each particle
        float m_Qthreshold;          //!< Dotproduct cutoff
        unsigned int m_Sthreshold;    //!< Solid-like num connections cutoff
        unsigned int m_num_particles; //!< Number of particles
        unsigned int m_l;  //!< Value of l for the spherical harmonic.

        //Pull cluster data into these
        unsigned int m_num_clusters;                                //!< Number of clusters found inthe last call to compute()
        boost::shared_array<unsigned int> m_cluster_idx;            //!< Cluster index determined for each particle
        std::vector< std::complex<float> > m_qldot_ij;     //!< All of the Qlmi dot Qlmj's computed
 //       unsigned int m_num_dotproducts;                             //!< size of M_qlmdot_ij;
        boost::shared_array<unsigned int> m_number_of_connections;  //!< Number of connections for each particle with dot product above Qthreshold
        boost::shared_array<unsigned int> m_number_of_neighbors;    //!< Number of neighbors for each particle (used for normalizing spherical harmonics);
        std::vector<unsigned int> m_number_of_shared_connections;  //!Stores number of shared neighbors for all ij pairs considered
    };

//! Exports all classes in this file to python
void export_SolLiqNear();

}; }; // end namespace freud::sol_liq_near

#endif // _SOL_LIQ_NEAR_H__
