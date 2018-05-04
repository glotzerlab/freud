// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is part of the freud project, released under the BSD 3-Clause License.

#include <complex>
#include <memory>
#include <ostream>
#include <tbb/tbb.h>

#include "box.h"
#include "VectorMath.h"
#include "NearestNeighbors.h"
#include "Index1D.h"

#ifndef _TRANS_ORDER_PARAMETER_H__
#define _TRANS_ORDER_PARAMETER_H__

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
                     const freud::locality::NeighborList *nlist,
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
        box::Box m_box;            //!< Simulation box where the particles belong
        float m_k;                 //!< Multiplier in the exponent
        unsigned int m_Np;         //!< Last number of points computed

        std::shared_ptr< std::complex<float> > m_dr_array;         //!< dr array computed
    };

}; }; // end namespace freud::order

#endif // _TRANS_ORDER_PARAMETER_H__
