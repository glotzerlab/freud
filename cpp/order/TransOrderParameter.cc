// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <complex>
#include <stdexcept>
#include <iostream>

#include "TransOrderParameter.h"

using namespace std;
using namespace tbb;

/*! \file TransOrderParameter.h
    \brief Compute the translational order parameter for each particle
*/

namespace freud { namespace order {

TransOrderParameter::TransOrderParameter(float k) 
    : OrderParameter<float>(k) {}

TransOrderParameter::~TransOrderParameter() {}

void TransOrderParameter::compute(const freud::locality::NeighborList* nlist,
                                  const freud::locality::NeighborQuery* points,
                                  freud::locality::QueryArgs qargs)
{
    computeGeneral(
    [] (vec3<float> &delta)
    {
        return complex<float>(delta.x, delta.y);
    }, 
    nlist, points, qargs);
}

}; }; // end namespace freud::order
