// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef CORRELATION_FUNCTION_H
#define CORRELATION_FUNCTION_H

#include <memory>
#include <tbb/tbb.h>

#include "Box.h"
#include "VectorMath.h"
#include "LinkCell.h"

/*! \file CorrelationFunction.h
    \brief Generic pairwise correlation functions.
*/

namespace freud { namespace density {

//! Computes the pairwise correlation function <p*q>(r) between two sets of points with associated values p and q.
/*! Two sets of points and two sets of values associated with those
    points are given. Computing the correlation function results in an
    array of the expected (average) product of all values at a given
    radial distance.

    The values of r at which to compute the correlation function are
    controlled by the rmax and dr parameters to the constructor. rmax
    determines the maximum r at which to compute the correlation
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
        const box::Box& getBox() const
            {
            return m_box;
            }

        //! Reset the PCF array to all zeros
        void reset();

        //! accumulate the correlation function
        void accumulate(const box::Box &box,
                        const freud::locality::NeighborList *nlist,
                        const vec3<float> *ref_points,
                        const T *ref_values,
                        unsigned int n_ref,
                        const vec3<float> *points,
                        const T *point_values,
                        unsigned int Np);

        //! \internal
        //! helper function to reduce the thread specific arrays into one array
        void reduceCorrelationFunction();

        //! Get a reference to the last computed rdf
        std::shared_ptr<T> getRDF();

        //! Get a reference to the bin counts array
        std::shared_ptr<unsigned int> getCounts()
            {
            reduceCorrelationFunction();
            return m_bin_counts;
            }

        //! Get a reference to the r array
        std::shared_ptr<float> getR()
            {
            return m_r_array;
            }

        unsigned int getNBins() const
            {
            return m_nbins;
            }

    private:
        box::Box m_box;                //!< Simulation box where the particles belong
        float m_rmax;                  //!< Maximum r at which to compute g(r)
        float m_dr;                    //!< Step size for r in the computation
        unsigned int m_nbins;          //!< Number of r bins to compute g(r) over
        unsigned int m_n_ref;          //!< number of reference particles
        unsigned int m_Np;             //!< number of check particles
        unsigned int m_frame_counter;  //!< number of frames calc'd
        bool m_reduce;                 //!< Whether arrays need to be reduced across threads

        std::shared_ptr<T> m_rdf_array;             //!< rdf array computed
        std::shared_ptr<unsigned int> m_bin_counts; //!< bin counts that go into computing the rdf array
        std::shared_ptr<float> m_r_array;           //!< array of r values where the rdf is computed
        tbb::enumerable_thread_specific<unsigned int *> m_local_bin_counts;
        tbb::enumerable_thread_specific<T *> m_local_rdf_array;
    };

}; }; // end namespace freud::density

#endif // CORRELATION_FUNCTION_H
