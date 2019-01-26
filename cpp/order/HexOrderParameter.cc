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

HexOrderParameter::HexOrderParameter(float rmax,
                                     unsigned int k,
                                     unsigned int n)
    : m_box(box::Box()), m_k(k), m_Np(0)
    {
    }

HexOrderParameter::~HexOrderParameter()
    {
    }

void HexOrderParameter::compute(box::Box& box,
                                const freud::locality::NeighborList *nlist,
                                const vec3<float> *points,
                                unsigned int Np)
    {
    // Compute the cell list
    m_box = box;

    nlist->validate(Np, Np);
    const size_t *neighbor_list(nlist->getNeighbors());

    // Reallocate the output array if it is not the right size
    if (Np != m_Np)
        {
        m_psi_array = std::shared_ptr<complex<float> >(new complex<float> [Np],
                std::default_delete<complex<float>[]>());
        }

    // Compute the order parameter
    parallel_for(blocked_range<size_t>(0,Np),
        [=] (const blocked_range<size_t>& r)
        {
        size_t bond(nlist->find_first_index(r.begin()));
        for (size_t i=r.begin(); i!=r.end(); ++i)
            {
            m_psi_array.get()[i] = 0;
            vec3<float> ref = points[i];

            for (; bond < nlist->getNumBonds() && neighbor_list[2*bond] == i;
                ++bond)
                {
                const size_t j(neighbor_list[2*bond + 1]);

                // Compute r between the two particles
                vec3<float> delta = m_box.wrap(points[j] - ref);

                float rsq = dot(delta, delta);
                if (rsq > 1e-6)
                    {
                    // Compute psi for neighboring particle
                    // (only constructed for 2d)
                    float psi_ij = atan2f(delta.y, delta.x);
                    m_psi_array.get()[i] += exp(complex<float>(0, m_k*psi_ij));
                    }
                }
            m_psi_array.get()[i] /= complex<float>(m_k);
            }
        });
    // Save the last computed number of particles
    m_Np = Np;
    }

}; }; // end namespace freud::order
