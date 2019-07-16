// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <complex>
#include <stdexcept>
#include <iostream>

#include "NeighborComputeFunctional.h"
#include "TransOrderParameter.h"

using namespace std;
using namespace tbb;

/*! \file TransOrderParameter.h
    \brief Compute the translational order parameter for each particle
*/

namespace freud { namespace order {

TransOrderParameter::TransOrderParameter(float rmax, float k) : m_box(box::Box()), m_k(k), m_Np(0) {}

TransOrderParameter::~TransOrderParameter() {}

void TransOrderParameter::compute(const freud::locality::NeighborList* nlist,
                                  const freud::locality::NeighborQuery* points, freud::locality::QueryArgs qargs)
{
    // compute the cell list
    m_box = points->getBox();

    unsigned int Np = points->getNRef();
    // reallocate the output array if it is not the right size
    if (Np != m_Np)
    {
        m_dr_array = std::shared_ptr<complex<float>>(new complex<float>[Np],
                                                     std::default_delete<complex<float>[]>());
    }

    freud::locality::loopOverNeighborsPoint(points, points->getRefPoints(), Np, qargs, nlist, 
    [=](size_t i)
    {
        m_dr_array.get()[i] = 0; return 0;
    }, 
    [=](size_t i, size_t j, float distance, float weight, int data)
    {
        vec3<float> ref = points->getRefPoints()[i];
        // compute r between the two particles
        vec3<float> delta = m_box.wrap(points->getRefPoints()[j] - ref);

        // compute dr for neighboring particle(only constructed for 2d)
        m_dr_array.get()[i] += complex<float>(delta.x, delta.y);
    },
    [=](size_t i, int data)
    {
        m_dr_array.get()[i] /= complex<float>(m_k);
    });

    // save the last computed number of particles
    m_Np = Np;
}

}; }; // end namespace freud::order
