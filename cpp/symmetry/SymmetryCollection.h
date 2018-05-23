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
        //! Constructor, set default value maxL to 30
        SymmetryCollection(unsigned int maxL = 30);

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
        
        //! fill in Mlm table
        float measure(std::shared_ptr<float> Mlm, int type);


        // quat<float> initMirrorZ(vector<float> p); 

        // int searchSymmetry(bool perpendicular);





        //! Returns quaternion corresponding to the highest-symmetry axis
        quat<float> getHighestOrderQuat();

        //! Returns quaternions for all detected symmetry axes
        quat<float>* getOrderQuats();



        std::shared_ptr<float> getMlm();
        
        unsigned int getNP() {
            return m_Np;
        }

        unsigned int getMaxL() {
            return m_maxL;
        }
    private:
        box::Box m_box;
        unsigned int m_maxL;
        quat<float> m_symmetric_orientation;
        std::shared_ptr<float> m_Mlm;
        unsigned int m_Np;
        const int TOTAL = -1;
        const int AXIAL = 0;
        const int MIRROR = 1;
        const int TWOFold = 2;
        const int THREEFold = 3;
        const int FOURFold = 4;
        const int FIVEFold = 5;
        const int SIXFold = 6;
        const int EIGHTFold = 8;
        const int TENFold = 10;
        const int TWELVEFold = 12;
    };


}; }; // end namespace freud::symmetry

#endif // _SYMMETRY_COLLECTION_H__
