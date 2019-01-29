// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef TRANS_ORDER_PARAMETER_H
#define TRANS_ORDER_PARAMETER_H

#include <complex>
#include <memory>
#include <ostream>
#include <tbb/tbb.h>

#include "Box.h"
#include "VectorMath.h"
#include "NearestNeighbors.h"
#include "Index1D.h"

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
        TransOrderParameter(float rmax, float k=6);

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
        float m_k;                 //!< Normalization value (dr is divided by m_k)
        unsigned int m_Np;         //!< Last number of points computed

        std::shared_ptr< std::complex<float> > m_dr_array;         //!< dr array computed
    };

}; }; // end namespace freud::order

#endif // TRANS_ORDER_PARAMETER_H
