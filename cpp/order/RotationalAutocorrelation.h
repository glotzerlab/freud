// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef ROTATIONAL_AUTOCORRELATION_H
#define ROTATIONAL_AUTOCORRELATION_H

#include <complex>
#include <memory>
#include <ostream>
#include <tbb/tbb.h>

#include "VectorMath.h"
#include "HOOMDMath.h"
#include "Index1D.h"

/*! \file RotationalAutocorrelation.h
    \brief Compute the rotational autocorrelation function for a system
    against a reference set of orientations
*/

namespace freud { namespace order {

//! Compute the translational order parameter for a set of points
/*!
*/
class RotationalAutocorrelationFunction
    {
    public:
        //! Constructor
        RotationalAutocorrelationFunction(int l);

        //! Destructor
        ~RotationalAutocorrelationFunction();

        //! Get the quantum number l used in calculations
        unsigned int getL()
            {
            return m_l;
            }

        //! Compute the Rotational Autocorrelation Function
        void compute(
                const quat<float> *ref_ors,
                const quat<float> *ors,
                unsigned int Np);

        //! Get a reference to the last computed dr
        std::shared_ptr< std::complex<float> > getDr()
            {
            return m_dr_array;
            }

        unsigned int getNP()
            {
            return m_Np;
            }

    private:
        float m_l;                 //!< Order of the hyperspherical harmonic
        unsigned int m_Np;         //!< Last number of points computed

        std::shared_ptr< std::complex<float> > m_dr_array;         //!< dr array computed
    };

}; }; // end namespace freud::order

#endif // ROTATIONAL_AUTOCORRELATION_H
