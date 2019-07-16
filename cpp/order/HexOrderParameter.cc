// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <complex>
#include <stdexcept>

#include "HexOrderParameter.h"

using namespace std;
using namespace tbb;

/*! \file HexOrderParameter.cc
    \brief Compute the hexatic order parameter for each particle.
*/

namespace freud { namespace order {

HexOrderParameter::HexOrderParameter(unsigned int k)
    : OrderParameter<unsigned int>(k) {}

HexOrderParameter::~HexOrderParameter() {}

void HexOrderParameter::compute(const freud::locality::NeighborList* nlist,
                                const freud::locality::NeighborQuery* points,
                                freud::locality::QueryArgs qargs)
{
    computeGeneral(
    [this] (vec3<float> &delta)
    {
        float psi_ij = atan2f(delta.y, delta.x); 
        return exp(complex<float>(0, m_k * psi_ij));
    }, 
    nlist, points, qargs);
}

}; }; // end namespace freud::order
