// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is part of the Freud project, released under the BSD 3-Clause License.

#include <complex>
#include <stdexcept>

#include "SymmetricOrientation.h"

using namespace std;

/*! \file SymmetricOrientation.cc
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

SymmetricOrientation::~SymmetricOrientation()
    {
    }

quat<float> SymmetricOrientation::getSymmetricOrientation()
    {
    return m_symmetric_orientation;
    }

}; }; // end namespace freud::symmetry
