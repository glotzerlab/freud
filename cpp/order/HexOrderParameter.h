// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is part of the freud project, released under the BSD 3-Clause License.

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

#ifndef _HEX_ORDER_PARAMTER_H__
#define _HEX_ORDER_PARAMTER_H__

/*! \file HexOrderParameter.h
    \brief Compute the hexatic order parameter for each particle
*/

namespace freud { namespace order {

//! Compute the hexagonal order parameter for a set of points
/*!
*/
class HexOrderParameter
    {
    public:
        //! Constructor
        HexOrderParameter(float rmax, float k=6, unsigned int n=0);

        //! Destructor
        ~HexOrderParameter();

        //! Get the simulation box
        const box::Box& getBox() const
            {
            return m_box;
            }

        //! Compute the hex order parameter
        void compute(box::Box& box,
                     const freud::locality::NeighborList *nlist,
                     const vec3<float> *points,
                     unsigned int Np);

        //! Get a reference to the last computed psi
        std::shared_ptr< std::complex<float> > getPsi()
            {
            return m_psi_array;
            }

        unsigned int getNP()
            {
            return m_Np;
            }

        float getK()
            {
            return m_k;
            }

    private:
        box::Box m_box;            //!< Simulation box where the particles belong
        float m_k;                 //!< Multiplier in the exponent
        unsigned int m_Np;         //!< Last number of points computed

        std::shared_ptr< std::complex<float> > m_psi_array;  //!< psi array computed
    };

}; }; // end namespace freud::order

#endif // _HEX_ORDER_PARAMTER_H__
