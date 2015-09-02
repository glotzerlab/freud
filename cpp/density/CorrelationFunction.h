#include <boost/shared_array.hpp>

#include "HOOMDMath.h"
#include "VectorMath.h"

#include "LinkCell.h"
#include "trajectory.h"

#include <tbb/tbb.h>

#ifndef _CORRELATIONFUNCTION_H__
#define _CORRELATIONFUNCTION_H__

/*! \file CorrelationFunction.cc
    \brief Generic pairwise correlation functions
*/

namespace freud { namespace density {

//! Computes the pairwise correlation function <p*q>(r) between two sets of points with associated values p and q.
/*! Two sets of points and two sets of values associated with those
    points are given. Computing the correlation function results in an
    array of the expected (average) product of all values at a given
    radial distance.

    The values of r to compute the correlation function at are
    controlled by the rmax and dr parameters to the constructor. rmax
    determins the maximum r at which to compute the correlation
    function and dr is the step size for each bin.

    <b>2D:</b><br>
    CorrelationFunction properly handles 2D boxes. As with everything
    else in freud, 2D points must be passed in as 3 component vectors
    x,y,0. Failing to set 0 in the third component will lead to
    undefined behavior.

    <b>Self-correlation:</b><br>
    It is often the case that we wish to compute the correlation
    function of a set of points with itself. If given the same arrays
    for both points and ref_points, we omit accumulating the
    self-correlation value in the first bin.

*/
template<typename T>
class CorrelationFunction
    {
    public:
        //! Constructor
        CorrelationFunction(float rmax, float dr);

        //! Destructor
        ~CorrelationFunction();

        //! Get the simulation box
        const trajectory::Box& getBox() const
            {
            return m_box;
            }

        //! Reset the PCF array to all zeros
        void resetCorrelationFunction();

        //! Python wrapper for reset method
        void resetCorrelationFunctionPy()
            {
            resetCorrelationFunction();
            }

        //! accumulate the correlation function
        void accumulate(const trajectory::Box &box,
                        const vec3<float> *ref_points,
                        const T *ref_values,
                        unsigned int Nref,
                        const vec3<float> *points,
                        const T *point_values,
                        unsigned int Np);

        // //! Python wrapper for accumulate
        // void accumulatePy(trajectory::Box& box,
        //                   boost::python::numeric::array ref_points,
        //                   boost::python::numeric::array ref_values,
        //                   boost::python::numeric::array points,
        //                   boost::python::numeric::array point_values);

        // //! Python wrapper for compute
        // void computePy(trajectory::Box& box,
        //                boost::python::numeric::array ref_points,
        //                boost::python::numeric::array ref_values,
        //                boost::python::numeric::array points,
        //                boost::python::numeric::array point_values);

        //! \internal
        //! helper function to reduce the thread specific arrays into the boost array
        void reduceCorrelationFunction();

        //! Get a reference to the last computed rdf
        boost::shared_array<T> getRDF();

        //! Get a reference to the bin counts array
        boost::shared_array<unsigned int> getCounts()
            {
            return m_bin_counts;
            }

        //! Get a reference to the r array
        boost::shared_array<float> getR()
            {
            return m_r_array;
            }

        unsigned int getNBins() const
            {
            return m_nbins;
            }

        // //! Python wrapper for getRDF() (returns a copy)
        // boost::python::numeric::array getRDFPy();
        //     // {
        //     // T *arr = m_rdf_array.get();
        //     // return num_util::makeNum(arr, m_nbins);
        //     // }

        // //! Python wrapper for getCounts() (returns a copy)
        // boost::python::numeric::array getCountsPy();
        //     // {
        //     // unsigned int *arr = m_bin_counts.get();
        //     // return num_util::makeNum(arr, m_nbins);
        //     // }

        // //! Python wrapper for getR() (returns a copy)
        // boost::python::numeric::array getRPy()
        //     {
        //     float *arr = m_r_array.get();
        //     return num_util::makeNum(arr, m_nbins);
        //     }

    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        float m_rmax;                     //!< Maximum r at which to compute g(r)
        float m_dr;                       //!< Step size for r in the computation
        locality::LinkCell* m_lc;          //!< LinkCell to bin particles for the computation
        unsigned int m_nbins;             //!< Number of r bins to compute g(r) over
        unsigned int m_Nref;                  //!< number of reference particles
        unsigned int m_Np;                  //!< number of check particles
        unsigned int m_frame_counter;       //!< number of frames calc'd

        boost::shared_array<T> m_rdf_array;         //!< rdf array computed
        boost::shared_array<unsigned int> m_bin_counts; //!< bin counts that go into computing the rdf array
        boost::shared_array<float> m_r_array;           //!< array of r values that the rdf is computed at
        tbb::enumerable_thread_specific<unsigned int *> m_local_bin_counts;
        tbb::enumerable_thread_specific<T *> m_local_rdf_array;
    };

// /*! \internal
//     \brief Template function to check the type of a given correlation
//         function value array. Should be specialized for its argument.
// */
// template<typename T>
// void checkCFType(boost::python::numeric::array values);

#include "CorrelationFunction.cc"

}; }; // end namespace freud::density

#endif // _CORRELATIONFUNCTION_H__
