// Copyright (c) 2010-2017 The Regents of the University of Michigan
// This file is part of the Freud project, released under the BSD 3-Clause License.

#include "SymmetricOrientation.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include <stdexcept>
#include <complex>

using namespace std;
using namespace tbb;

/*! \file SymmetricOrientation.h
    \brief Compute the symmetric orientation.
*/

namespace freud { namespace symmetry {

SymmetricOrientation::SymmetricOrientation()
    {
    m_symmetric_orientation.s = 1;
    m_symmetric_orientation.v.x = 0;
    m_symmetric_orientation.v.y = 0;
    m_symmetric_orientation.v.z = 0;
    }

//SymmetricOrientation::~SymmetricOrientation()
//    {
//    }

quat<float> SymmetricOrientation::getSymmetricOrientation()
    {
    return m_symmetric_orientation;
    }

}; }; // end namespace freud::symmetry
