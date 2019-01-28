// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef ROTATIONAL_AUTOCORRELATION_FUNCTION_H
#define ROTATIONAL_AUTOCORRELATION_FUNCTION_H

#include <complex>
#include <memory>
#include <cassert>
#include <tbb/tbb.h>

#include "VectorMath.h"
#include "HOOMDMath.h"
#include "Index1D.h"

/*! \file RotationalAutocorrelationFunction.h
    \brief Defines the RotationalAutocorrelationFunction class, which computes
    the rotational autocorrelation function for a system against a reference
    set of orientations.
*/

namespace freud { namespace order {

//! Convert a quaternion to complex coordinates.
/*! The set of quaternions are isomorphic to the special unitary group of
 *  degree 2 SU(2), which forms a double cover of the 3D rotation group SO(3).
 *  SU(2) is also diffeomorphic to the 3-sphere S3 (see
 *  https://en.wikipedia.org/wiki/Special_unitary_group#Diffeomorphism_with_S3,
 *  for example), meaning that we can represent quaternions in terms of two
 *  complex numbers that map out hyperspherical coordinates in 3 dimensions.
 *  This function generates that mapping.
 */
std::pair<std::complex<float>, std::complex<float> > quat_to_greek(const quat<float> &q);

//! Compute a hyperspherical harmonic.
/*! The hyperspherical harmonic function is a generalization of spherical
 *  harmonics from the 2-sphere to the 3-sphere. For details, see Harmonic
 *  functions and matrix elements for hyperspherical quantum field models
 *  (https://doi.org/10.1063/1.526210).
 */
std::complex<float> hypersphere_harmonic(const std::complex<float> xi, std::complex<float> zeta,
                                          const int l, const int m1, const int m2);

//! Compute the rotational autocorrelation function for a set of orientations.
/*! The desired autocorrelation function is the rotational analog of the
 *  dynamic structure factor, which provides information on the dynamcs of
 *  systems of points. Calculating this quantity requires a generalization of
 *  the Fourier transform to a different domain, namely the rotation group
 *  SO(3). This computation can be performed using a hyperspherical coordinate
 *  representation of the rotations. For details, see "Design rules for
 *  engineering colloidal plastic crystals of hard polyhedra – phase behavior
 *  and directional entropic forces" (in preparation).
 */
class RotationalAutocorrelationFunction
    {
    public:
        //! Explicit default constructor for Cython.
        RotationalAutocorrelationFunction() {}

        //! Constructor
        RotationalAutocorrelationFunction(int l) : m_l(l), m_Np(0), m_Ft(0) {}

        //! Destructor
        ~RotationalAutocorrelationFunction() {}

        //! Get the quantum number l used in calculations.
        unsigned int getL()
            {
            return m_l;
            }

        //! Get the number of orientations used in the last call to compute.
        unsigned int getNP()
            {
              return m_Np;
            }

        //! Get a reference to the last computed global angle array.
        std::shared_ptr<std::complex <float> > getRAArray()
            {
            return m_RA_array;
            }

        //! Get a reference to the last computed value of the rotational autocorrelation function.
        float getRotationalAutocorrelationFunction()
            {
            return m_Ft;
            }

        //! Compute the rotational autocorrelation function.
        /*! This function loops over all provided orientations and reference
         *  orientations and computes their hyperspherical harmonics for the
         *  desired range of quantum numbers. For each orientation/reference
         *  pair, the autocorrelation value is computed as the inner product of
         *  these two hyperspherical harmonics. The value of the autocorrelation
         *  for the whole system is then the average of the real parts of the
         *  autocorrelation for the whole system.
         */
        void compute(const quat<float> *ref_ors, const quat<float> *ors, unsigned int Np);

    private:
        int m_l;                   //!< Order of the hyperspherical harmonic
        unsigned int m_Np;         //!< Last number of points computed
        float m_Ft;                //!< Real value of calculated R.A. function

        std::shared_ptr< std::complex<float> > m_RA_array; //!< Array of RA values per particle
    };

}; }; // end namespace freud::order

#endif // ROTATIONAL_AUTOCORRELATION_FUNCTION_H
