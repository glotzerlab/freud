// Copyright (c) 2010-2016 The Regents of the University of Michigan
// This file is part of the Freud project, released under the BSD 3-Clause License.

#include <tbb/tbb.h>
#include <ostream>
#include <complex>

// work around nasty issue where python #defines isalpha, toupper, etc....
#undef __APPLE__
#include <Python.h>
#define __APPLE__

#include <memory>

#include "HOOMDMath.h"
#include "VectorMath.h"

#include "NearestNeighbors.h"
#include "box.h"
#include "Index1D.h"

#ifndef _TRANS_ORDER_PARAMTER_H__
#define _TRANS_ORDER_PARAMTER_H__

/*! \file TransOrderParameter.h
    \brief Compute the translational order parameter for each particle
*/

namespace freud { namespace order {

//! Compute the translational order parameter for a set of points
/*!
*/
class TransOrderParameter
    {
    public:
        //! Constructor
        TransOrderParameter(float rmax, float k=6, unsigned int n=0);

        //! Destructor
        ~TransOrderParameter();

        //! Get the simulation box
        const box::Box& getBox() const
            {
            return m_box;
            }

        //! Compute the translational order parameter
        void compute(box::Box& box,
                     const vec3<float> *points,
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
        box::Box m_box;            //!< Simulation box the particles belong in
        float m_rmax;                     //!< Maximum r at which to determine neighbors
        float m_k;                        //!< Multiplier in the exponent
        locality::NearestNeighbors *m_nn;          //!< Nearest Neighbors for the computation
        unsigned int m_Np;                //!< Last number of points computed

        std::shared_ptr< std::complex<float> > m_dr_array;         //!< dr array computed
    };

}; }; // end namespace freud::order

#endif // _TRANS_ORDER_PARAMTER_H__
