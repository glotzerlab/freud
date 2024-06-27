// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "LocalDensity.h"
#include "NeighborComputeFunctional.h"
#include "NeighborBond.h"
#include <tbb/enumerable_thread_specific.h>
#include "<vector>"

/*! \file LocalDensity.cc
    \brief Routines for computing local density around a point.
*/

namespace freud { namespace density {

LocalDensity::LocalDensity(float r_max, float diameter)
    : m_box(box::Box()), m_r_max(r_max), m_diameter(diameter)
{
    if (r_max <= 0)
    {
        throw std::invalid_argument("LocalDensity requires r_max to be positive.");
    }

    if (diameter < 0)
    {
        throw std::invalid_argument("LocalDensity requires diameter to be non-negative.");
    }
}

void LocalDensity::compute(const freud::locality::NeighborQuery* neighbor_query,
                           const vec3<float>* query_points, unsigned int n_query_points,
                           const freud::locality::NeighborList* nlist, freud::locality::QueryArgs qargs)
{
    m_box = neighbor_query->getBox();
    using BondVector = tbb::enumerable_thread_specific<std::vector<NeighborBond>>;
    BondVector new_bonds;

    m_density_array.prepare(n_query_points);
    m_num_neighbors_array.prepare(n_query_points);

    const float area = M_PI * m_r_max * m_r_max;
    const float volume = static_cast<float>(4.0 / 3.0 * M_PI) * m_r_max * m_r_max * m_r_max;
    // compute the local density
    freud::locality::loopOverNeighborsIterator(
        neighbor_query, query_points, n_query_points, qargs, nlist,
        [&](size_t i, const std::shared_ptr<freud::locality::NeighborPerPointIterator>& ppiter) {
            float weight;
            float num_neighbors = 0;
            for (freud::locality::NeighborBond nb = ppiter->next(); !ppiter->end(); nb = ppiter->next())
            {
                // count particles that are fully in the r_max sphere
                if (nb.getDistance() < (m_r_max - m_diameter / float(2.0)))
                {
                    weight = float(1.0);
                }
                else
                {
                    // partially count particles that intersect the r_max sphere
                    // this is not particularly accurate for a single particle, but works well on average for
                    // lots of them. It smooths out the neighbor count distributions and avoids noisy spikes
                    // that obscure data
                    weight = float(1.0) + (m_r_max - (nb.getDistance() + m_diameter / float(2.0))) / m_diameter;
                }
                new_bonds.emplace_back(i, nb.getPointIdx(),nb.getDistance(),weight,nb.getVector())
                num_neighbors += weight;
                m_num_neighbors_array[i] = num_neighbors;
                if (m_box.is2D())
                {
                    // local density is area of particles divided by the area of the circle
                    m_density_array[i] = m_num_neighbors_array[i] / area;
                }
                else
                {
                    // local density is volume of particles divided by the volume of the sphere
                    m_density_array[i] = m_num_neighbors_array[i] / volume;
                }
            }
        });
        tbb::flattened2d<BondVector> flat_density_bonds = tbb::flatten2d(new_bonds);
        std::vector<NeighborBond> density_bonds(flat_density_bonds.begin(), flat_density_bonds.end());

        m_density_nlist = std::make_shared<NeighborList>(density_bonds);
}

}; }; // end namespace freud::density
