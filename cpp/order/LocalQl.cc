// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is part of the freud project, released under the BSD 3-Clause License.

#include "LocalQl.h"

using namespace std;

/*! \file LocalQl.cc
    \brief Compute a Ql per particle
*/

namespace freud { namespace order {

LocalQl::LocalQl(const box::Box& box, float rmax, unsigned int l, float rmin) : Steinhardt(box, rmax, l, rmin) {}

}; }; // end namespace freud::order
