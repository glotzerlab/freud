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

int compareInts(const void * a, const void * b);

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
        Bootstrap(const unsigned int nBootstrap, boost::python::numeric::array data_array);

        //! Destructor
        ~Bootstrap();

        void AnalyzeBootstrap(boost::shared_array<unsigned int> *bootstrap_array,
                              boost::shared_array<float> *avg_array,
                              boost::shared_array<float> *std_array,
                              boost::shared_array<float> *err_array,
                              std::vector<unsigned int> *cum_array);

        //! Python wrapper for getPCF() (returns a copy)
        boost::python::numeric::array getBootstrapPy()
            {
            unsigned int *arr = m_bootstrap_array.get();
            return num_util::makeNum(arr, m_nBootstrap * m_arrSize);
            }

        //! Python wrapper for getPCF() (returns a copy)
        boost::python::numeric::array getAVGPy()
            {
            float *arr = m_avg_array.get();
            return num_util::makeNum(arr, m_arrSize);
            }

        //! Python wrapper for getPCF() (returns a copy)
        boost::python::numeric::array getSTDPy()
            {
            float *arr = m_std_array.get();
            return num_util::makeNum(arr, m_arrSize);
            }

        //! Python wrapper for getPCF() (returns a copy)
        boost::python::numeric::array getERRPy()
            {
            float *arr = m_err_array.get();
            return num_util::makeNum(arr, m_arrSize);
            }

        //! Compute the bootstrap analysis
        // will handle both the computation and analysis
        void compute();

        //! Python wrapper for compute
        void computePy();

    private:
        const unsigned int  m_nBootstrap;    //!< number of bootstrap arrays to compute
        std::vector<unsigned int> *m_data_array; //!< data array
        std::vector<unsigned int> *m_cum_array; //!< cumulative data array
        boost::shared_array<unsigned int> m_bootstrap_array;         //!< array of pcf computed
        boost::shared_array<float> m_avg_array;         //!< array of pcf computed
        boost::shared_array<float> m_std_array;         //!< array of pcf computed
        boost::shared_array<float> m_err_array;         //!< array of pcf computed
        unsigned int  m_nPoints;    //!< number of points to populate the bootstrap arrays with
        unsigned int  m_arrSize;    //!< number of points to populate the bootstrap arrays with
    };

/*! \internal
    \brief Exports all classes in this file to python
*/
void export_Bootstrap();

}; }; // end namespace freud::pmft

#endif // _Bootstrap_H__
