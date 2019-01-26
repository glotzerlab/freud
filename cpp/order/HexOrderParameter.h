// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef HEX_ORDER_PARAMETER_H
#define HEX_ORDER_PARAMETER_H

#include <complex>
#include <memory>
#include <ostream>
#include <tbb/tbb.h>

#include "Box.h"
#include "VectorMath.h"
#include "NearestNeighbors.h"
#include "Index1D.h"

/*! \file HexOrderParameter.h
    \brief Compute the hexatic order parameter for each particle.
*/

namespace freud { namespace order {

//! Compute the hexagonal order parameter for a set of points
/*!
*/
class HexOrderParameter
    {
    public:
        //! Constructor
        HexOrderParameter(float rmax, unsigned int k=6, unsigned int n=0);

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

        unsigned int getK()
            {
            return m_k;
            }

    private:
        box::Box m_box;            //!< Simulation box where the particles belong
        unsigned int m_k;          //!< Multiplier in the exponent
        unsigned int m_Np;         //!< Last number of points computed

        std::shared_ptr< std::complex<float> > m_psi_array;  //!< psi array computed
    };

}; }; // end namespace freud::order

#endif // HEX_ORDER_PARAMETER_H
