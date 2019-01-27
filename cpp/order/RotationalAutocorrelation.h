// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef ROTATIONAL_AUTOCORRELATION_FUNCTION_H
#define ROTATIONAL_AUTOCORRELATION_FUNCTION_H

#include <complex>
#include <memory>
#include <iostream>
#include <ostream>
#include <tbb/tbb.h>

#include "VectorMath.h"
#include "HOOMDMath.h"
#include "Index1D.h"

/*! \file RotationalAutocorrelationFunction.h
    \brief Compute the rotational autocorrelation function for a system
    against a reference set of orientations
*/

namespace freud { namespace order {

//! Setting up a couple of functions that are used in calculation
std::pair<std::complex<float>, std::complex<float> > quat_to_greek(const quat<float> &q);

std::complex<float> hypersphere_harmonic(const std::complex<float> xi, std::complex<float> zeta,
                                          const int l, const int m1, const int m2);

//! Compute the translational order parameter for a set of points
/*!
*/
class RotationalAutocorrelationFunction
    {
    public:
        //! Constructor
        RotationalAutocorrelationFunction()
        {}
        RotationalAutocorrelationFunction(int l); //:m_l(l)
//            {
//            if (m_l < 2)
//                throw std::invalid_argument("l must be two or greater!");
//            }

        //! Destructor
        ~RotationalAutocorrelationFunction();

        //! Get the quantum number l used in calculations
        unsigned int getL()
            {
            return m_l;
          };

        unsigned int getNP()
            {
              return m_Np;
            };

      //! Get a reference to the last computed global angle array
      std::shared_ptr<std::complex <float> > getRAArray()
          {
          return m_RA_array;
          };

      float getRotationalAutocorrelationFunction()
          {
              std::cout << "In the get function, have" << m_Ft << std::endl;
          return m_Ft;
          };


        //! Compute the Rotational Autocorrelation Function
        void compute(
                const quat<float> *ref_ors,
                const quat<float> *ors,
                unsigned int Np);

    private:
        int m_l;                 //!< Order of the hyperspherical harmonic
        unsigned int m_Np;         //!< Last number of points computed
        float m_Ft;                //!< Real value of calculated R.A. function

        std::shared_ptr< std::complex<float> > m_RA_array; //!< Array of RA values per particle
    };

}; }; // end namespace freud::order

#endif // ROTATIONAL_AUTOCORRELATION_FUNCTION_H
