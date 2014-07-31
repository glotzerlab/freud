#include <boost/python.hpp>
#include <boost/shared_array.hpp>

// probably don't need these...

#include "LinkCell.h"
#include "num_util.h"
#include "trajectory.h"

#ifndef _Bootstrap_H__
#define _Bootstrap_H__

/*! \file Bootstrap.h
    \brief Routines for computing bootstrap analysis
*/

namespace freud { namespace bootstrap {

// update this
//! Computes the RDF (g(r)) for a given set of points
/*! A given set of reference points is given around which the RDF is computed and averaged in a sea of data points.
    Computing the RDF results in an rdf array listing the value of the RDF at each given r, listed in the r array.

    The values of r to compute the rdf at are controlled by the rmax and dr parameters to the constructor. rmax
    determins the maximum r at which to compute g(r) and dr is the step size for each bin.

    <b>2D:</b><br>
    RDF properly handles 2D boxes. As with everything else in freud, 2D points must be passed in as
    3 component vectors x,y,0. Failing to set 0 in the third component will lead to undefined behavior.
*/
class Bootstrap
    {
    public:
        //! Constructor
        Bootstrap(const unsigned int nBootstrap, const unsigned int nPoints, const unsigned int arrSize);

        //! Destructor
        ~Bootstrap();

        int compareInts(const void * a, const void * b);

        void AnalyzeBootstrap(unsigned int *bootstrapArray,
                              unsigned int *bootstrapAVG,
                              unsigned int *bootstrapSTD,
                              unsigned int *bootstrapRatio,
                              unsigned int *dataCum);

        //! Compute the bootstrap analysis
        // will handle both the computation and analysis
        void compute(unsigned int *bootstrapArray,
                     float *bootstrapAVG,
                     float *bootstrapSTD,
                     float bootstrapRatio,
                     unsigned int *dataCum);

        //! Python wrapper for compute
        void computePy(boost::python::numeric::array bootstrapArray,
                       boost::python::numeric::array bootstrapAVG,
                       boost::python::numeric::array bootstrapSTD,
                       boost::python::numeric::array bootstrapRatio,
                       boost::python::numeric::array dataCum);

    private:
        unsigned int  m_nBootstrap;    //!< number of bootstrap arrays to compute
        unsigned int  m_nPoints;    //!< number of points to populate the bootstrap arrays with
        unsigned int  m_arrSize;    //!< number of points to populate the bootstrap arrays with
    };

/*! \internal
    \brief Exports all classes in this file to python
*/
void export_Bootstrap();

}; }; // end namespace freud::pmft

#endif // _Bootstrap_H__
