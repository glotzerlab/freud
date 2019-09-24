// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef CORRELATION_FUNCTION_H
#define CORRELATION_FUNCTION_H

#include "BondHistogramCompute.h"
#include "Box.h"
#include "Histogram.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "ThreadStorage.h"
#include "VectorMath.h"
#include "ManagedArray.h"

/*! \file CorrelationFunction.h
    \brief Generic pairwise correlation functions.
*/

namespace freud { namespace density {

//! Computes the pairwise correlation function <p*q>(r) between two sets of points with associated values p
//! and q.
/*! Two sets of points and two sets of values associated with those
    points are given. Computing the correlation function results in an
    array of the expected (average) product of all values at a given
    radial distance.

    The values of r at which to compute the correlation function are
    controlled by the r_max and dr parameters to the constructor. r_max
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
class CorrelationFunction : public locality::BondHistogramCompute
{
public:
    //! Constructor
    CorrelationFunction(float r_max, float dr);

    //! Destructor
    ~CorrelationFunction() {}

    //! Get the simulation box
    const box::Box& getBox() const
    {
        return m_box;
    }

    //! Reset the PCF array to all zeros
    virtual void reset();

    //! accumulate the correlation function
    void accumulate(const freud::locality::NeighborQuery* neighbor_query, const T* values,
                    const vec3<float>* query_points, const T* query_values,
                    unsigned int n_query_points, const freud::locality::NeighborList* nlist,
                    freud::locality::QueryArgs qargs);

    //! \internal
    //! helper function to reduce the thread specific arrays into one array
    void reduce();

    //! Get a reference to the last computed rdf
    const util::ManagedArray<T> &getRDF()
    {
        return reduceAndReturn(m_correlation_function.getBinCounts());
    }

    //! Get a reference to the r array
    const util::ManagedArray<float> &getR()
    {
        return m_r_array;
    }

    unsigned int getNBins() const
    {
        return m_nbins;
    }

private:
    float m_r_max;                 //!< Maximum r at which to compute g(r)
    float m_dr;                   //!< Step size for r in the computation
    unsigned int m_nbins;         //!< Number of r bins to compute g(r) over

    util::ManagedArray<float> m_r_array;           //!< array of r values where the rdf is computed

    // Typedef thread local histogram type for use in code.
    typedef typename util::Histogram<T>::ThreadLocalHistogram CFThreadHistogram;

    util::Histogram<T> m_correlation_function; //!< The correlation function
    CFThreadHistogram m_local_correlation_function; //!< Thread local copy of the correlation function
};

}; }; // end namespace freud::density

#endif // CORRELATION_FUNCTION_H
