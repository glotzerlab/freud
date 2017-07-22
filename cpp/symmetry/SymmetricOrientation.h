// Copyright (c) 2010-2017 The Regents of the University of Michigan
// This file is part of the Freud project, released under the BSD 3-Clause License.

#include <tbb/tbb.h>
#include <ostream>

// work around nasty issue where python #defines isalpha, toupper, etc....
#undef __APPLE__
#include <Python.h>
#define __APPLE__

#include <memory>

#include "HOOMDMath.h"
#include "VectorMath.h"

#ifndef _SYMMETRIC_ORIENTATION_H__
#define _SYMMETRIC_ORIENTATION_H__

/*! \file SymmetricOrientation.h
    \brief Compute the symmetric orientation
*/

namespace freud { namespace symmetry {

//! Compute the symmetric orientation
/*!
*/
class SymmetricOrientation
    {
    public:
        //! Constructor
        SymmetricOrientation();

        //! Destructor
        //~SymmetricOrientation();

        //! Get the symmetric orientation
        quat<float> getSymmetricOrientation();

    private:
        quat<float> m_symmetric_orientation;
    };

}; }; // end namespace freud::symmetry

#endif // _SYMMETRIC_ORIENTATION_H__
