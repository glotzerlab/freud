#include <ostream>

//work around nasty issue where python #defines isalpha, touppper, etc...
#undef __APPLE__
#include <Python.h>
#define __APPLE__

#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#include "num_util.h"

#include <vector>
#include <queue>

#ifndef _PhaseCluster_H__
#define _PhaseCluster_H__

namespace freud { namespace cluster{
//! Compute the phase clusters
/*! Implement Carolyn L. Phillips's paper: Discovering crystals using shape 
    matching and machine learning. Soft Matter, 2013, 9, 8552.
*/



class PhaseCluster
{
public:
    //! Constructor, use default b,z,X,f
    PhaseCluster();

    //! Constructor, give user defined b,z,X,f
    PhaseCluster(float bVal, float zVal, float XVal, float fVal);

    //! Destructor
    ~PhaseCluster();

    class compare
    {
    public:
        bool operator()(std::vector< float > a, std::vector< float > b)
        {
            return a[1] > b[1];
        }
    };

    //! return = arrayA - arrayB(each element)
    float* operateMinus(float *a, float *b);

    //! Calculate norm of a vector
    float getNorm(float *a);

    //! Calculate distance of two vectors
    float getDis(float *a, float *b);

    //! Divide RDF at local minimum, called in getRcut(...) function
    void divideLocMin(std::vector< std::vector< float > >& grData);


    //! Get cutOff radius of a RDF
    float getRcut(unsigned int numShell,
                  const std::vector< float >& r,
                  const std::vector< float >& gr);

    //! Python wrapper for getRcut
    float getRcutPy(unsigned int numShell,
                  boost::python::numeric::array r,
                  boost::python::numeric::array rdf);


    //! Get neighbor points indices, called in ExpandCluster(...)
    std::vector< int > regionQuery(float **pointSet, int pointInd, float epsilon);

    //! Search near points and expand cluster, called in DBSCAN(...)
    bool ExpandCluster(float **pointSet, int pointInd, int ClId, float epsilon, int minPts);

    //! Density Based Spatial Clustering of Applications with Noise algorithm
    void DBSCAN(float **pointSet, float epsilon, int minPts);

    //! Python wrapper for DBSCAN
    void DBSCANPy(boost::python::numeric::array pointSet, float epsilon, int minPts);






    //! Get core-distance of a point
    float setCoreDistance(float **pointSet, std::vector< int > neighbors, int pointInd, int minPts);


    //! called in ExpandClusterOrder(...)
    void updateOrderSeeds(float **pointSet, std::vector< int > neighbors, int pointInd, std::vector< int >& processVec, std::priority_queue< std::vector< float >, std::vector< std::vector< float > >, compare>& orderSeeds, float coreDist);

    //! Expand search cluster, called in OPTICS(...) function
    void ExpandClusterOrder(float **pointSet, int pointInd, float epsilon, int minPts, std::vector< int >& processVec);

    //! Ordering points to identify the clustering structure algorithm
    void OPTICS(float **pointSet, float epsilon, int minPts);

    //! Python wrapper for OPTICS
    void OPTICSPy(boost::python::numeric::array pointSet, float epsilon, int minPts);





    //! Get a reference to pointClusterID
    //boost::shared_array< float > getPointClusterID();

    //! Python wrapper for getPointClusterID() (returns a copy)
    boost::python::numeric::array getPointClusterIDPy();

    //! Get a reference to orderedPointIndex
    //boost::shared_array< float > getOrderedPointIndex();

    //! Python wrapper for getOrderedPointIndex() (returns a copy)
    boost::python::numeric::array getOrderedPointIndexPy();

    //! Get a reference to reachabilityDistance
    //boost::shared_array< float > getReachabilityDistance();

    //! Python wrapper for getReachabilityDistance() (returns a copy)
    boost::python::numeric::array getReachabilityDistancePy();

    //! Get a reference to coreDistance
    //boost::shared_array< float > getCoreDistance();

    //! Python wrapper for getCoreDistance() (returns a copy)
    boost::python::numeric::array getCoreDistancePy();


private:
    float b; //getRcut(..) function parameter. same with Carolyn's paper
    float z; //getRcut(..) function parameter
    float X;  //getRcut(..) function parameter
    float f;  //getRcut(..) function parameter

    int pointSetWidth; //pointSet column number
    int pointSetHeight; //pointSet row number

    std::vector< int > pointClusterID; //DBSCAN returns the cluster index for each point

    std::vector< int > orderedPointIndex; //OPTICS returns ordered points
    std::vector< float > reachabilityDistance; //OPTICS returns reachability distance for each point, same order with initial input pointSet
    std::vector< float > coreDistance; //OPTICS returns core distance for each point, same order with orderedPointIndex

};

//! \internal brief Exports all classes in this file to python
void export_PhaseCluster();

}; }; //end namespace freud::cluster

#endif // _PhaseCluster_H__

