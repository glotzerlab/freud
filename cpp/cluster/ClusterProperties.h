#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#include "HOOMDMath.h"
#include "num_util.h"
#include "trajectory.h"

#ifndef _CLUSTER_PROPERTIES_H__
#define _CLUSTER_PROPERTIES_H__

namespace freud { namespace cluster {

//! Computes properties of clusters
/*! Given a set of points and \a cluster_idx (from Cluster, or some other source), ClusterProperties determines
    the following properties for each cluster:
     - Center of mass
     - Gyration radius tensor
     
    m_cluster_com stores the computed center of mass for each cluster (properly handling periodic boundary conditions,
    of course). It is an array of float3's in c++. It is passed to python from getClusterCOMPy as an num_clusters x 3 
    numpy array.
    
    m_cluster_G stores a 3x3 G tensor for each cluster. Index cluster \a c, element \a j, \a i with the following:
    m_cluster_G[c*9 + j*3 + i]. The tensor is symmetric, so the choice of i and j are irrelevant. This is passed
    back to python as a num_clusters x 3 x 3 numpy array.
*/
class ClusterProperties
    {
    public:
        //! Constructor
        ClusterProperties(const trajectory::Box& box);
        
        //! Get the simulation box
        const trajectory::Box& getBox() const
            {
            return m_box;
            }
        
        //! Compute properties of the point clusters
        void computeProperties(const float3 *points,
                               const unsigned int *cluster_idx,
                               unsigned int Np);
        
        //! Python wrapper for computeProperties
        void computePropertiesPy(boost::python::numeric::array points,
                                 boost::python::numeric::array cluster_idx);
        
        //! Count the number of clusters found in the last call to computeProperties()
        unsigned int getNumClusters()
            {
            return m_num_clusters;
            }
        
        //! Get a reference to the last computed cluster_com
        boost::shared_array<float3> getClusterCOM()
            {
            return m_cluster_com;
            }
        
        //! Python wrapper for getClusterCOM() (returns a copy)
        boost::python::numeric::array getClusterCOMPy()
            {
            float *arr = (float*)m_cluster_com.get();
            std::vector<intp> dims(2);
            dims[0] = m_num_clusters;
            dims[1] = 3;
            return num_util::makeNum(arr, dims);
            }
        
        //! Get a reference to the last computec cluster_G
        boost::shared_array<float> getClusterG()
            {
            return m_cluster_G;
            }
            
        //!  Returns the cluster G tensors computed by the last call to computeProperties
        boost::python::object getClusterGPy()
            {
            float *arr = m_cluster_G.get();
            std::vector<intp> dims(3);
            dims[0] = m_num_clusters;
            dims[1] = 3;
            dims[2] = 3;
            return num_util::makeNum(arr, dims);
            }
        
    private:
        trajectory::Box m_box;                       //!< Simulation box the particles belong in
        unsigned int m_num_clusters;                 //!< Number of clusters found in the last call to computeProperties()
        
        boost::shared_array<float3> m_cluster_com;   //!< Center of mass computed for each cluster (length: m_num_clusters)
        boost::shared_array<float> m_cluster_G;      //!< Gyration tensor computed for each cluster (m_num_clusters x 3 x 3 array)
    };

/*! \internal
    \brief Exports all classes in this file to python 
*/
void export_ClusterProperties();

}; }; // end namespace freud::cluster

#endif // _CLUSTER_PROPERTIES_H__
