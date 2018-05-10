// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is part of the Freud project, released under the BSD 3-Clause License.

#include <memory>
#include <ostream>
#include <tbb/tbb.h>

#include "VectorMath.h"
#include "fsph/src/spherical_harmonics.hpp"
#include "LinkCell.h"

#ifndef _SYMMETRY_COLLECTION_H__
#define _SYMMETRY_COLLECTION_H__

/*! \file SymmetricOrientation.h
    \brief Compute the symmetric orientation
*/

namespace freud { namespace symmetry {

//! Compute the symmetric orientation
/*!
*/
class SymmetryCollection
    {
    public:
        //! Constructor
        SymmetryCollection();

        //! Destructor
        ~SymmetryCollection();

        //! Get the symmetric orientation
        quat<float> getSymmetricOrientation();

        //! Compute spherical harmonics from bond array
        void computeMlm(const box::Box& box,
                        const vec3<float> *points,
                        const freud::locality::NeighborList *nlist,
                        unsigned int Np);

        //! Compute symmetry axes
        void compute(const box::Box& box,
                     const vec3<float> *points,
                     const freud::locality::NeighborList *nlist,
                     unsigned int Np);

        //! Returns quaternion corresponding to the highest-symmetry axis
        quat<float> getHighestOrderQuat();

        //! Returns quaternions for all detected symmetry axes
        quat<float>* getOrderQuats();

        std::shared_ptr<std::complex<float>> getMlm();
        unsigned int getNP() {
            return m_Np;
        }

    private:
        box::Box m_box;
        const int MAXL = 30;
        quat<float> m_symmetric_orientation;
        std::shared_ptr<std::complex<float>> m_Mlm;
        unsigned int m_Np;
    };

}; }; // end namespace freud::symmetry

#endif // _SYMMETRY_COLLECTION_H__
