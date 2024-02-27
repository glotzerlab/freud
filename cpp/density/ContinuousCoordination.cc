// Copyright (c) 2010-2023 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <cmath>
#include <functional>
#include <numeric>

#include "ContinuousCoordination.h"
#include "NeighborComputeFunctional.h"

/*! \file ContinuousCoordination.cc
    \brief Routines for computing local density around a point.
*/

namespace freud { namespace density {

ContinuousCoordination::ContinuousCoordination(std::vector<float> powers, bool compute_log, bool compute_exp)
    : m_powers(std::move(powers)), m_compute_exp(compute_exp), m_compute_log(compute_log)
{}

void ContinuousCoordination::compute(const freud::locality::Voronoi* voronoi,
                                     const freud::locality::NeighborList* nlist, bool is2D)
{
    size_t num_points = nlist->getNumQueryPoints();
    m_coordination.prepare({num_points, getNumberOfCoordinations()});
    const auto& volumes = voronoi->getVolumes();
    const auto& num_neighbors = nlist->getCounts();
    // 2 for triangles 3 for pyramids
    const float volume_prefactor = is2D ? 2.0 : 3.0;
    freud::locality::loopOverNeighborListIterator(
        nlist,
        [&](size_t particle_index, const std::shared_ptr<freud::locality::NeighborPerPointIterator>& ppiter) {
            // 1/2 comes from the distance vector since we want to measure from the pyramid
            // base to the center.
            const float prefactor
                = 1.0F / (volume_prefactor * 2.0F * static_cast<float>(volumes[particle_index]));
            std::vector<float> i_volumes;
            for (freud::locality::NeighborBond nb = ppiter->next(); !ppiter->end(); nb = ppiter->next())
            {
                i_volumes.emplace_back(prefactor * nb.getWeight() * nb.getDistance());
            }
            size_t coordination_number {0};
            float num_neighbors_i {static_cast<float>(num_neighbors[particle_index])};
            for (size_t k {0}; k < m_powers.size(); ++k)
            {
                float coordination = std::transform_reduce(
                    i_volumes.begin(), i_volumes.end(), 0.0F, std::plus<>(),
                    [*this, k](const auto& volume) { return std::pow(volume, this->m_powers[k]); });
                m_coordination(particle_index, coordination_number++)
                    = std::pow(num_neighbors_i, 2.0F - m_powers[k]) / coordination;
            }
            if (m_compute_log)
            {
                float coordination
                    = std::transform_reduce(i_volumes.begin(), i_volumes.end(), 0.0F, std::plus<>(), logf);
                m_coordination(particle_index, coordination_number++)
                    = -coordination / std::log(num_neighbors_i);
            }
            if (m_compute_exp)
            {
                m_coordination(particle_index, coordination_number) = std::transform_reduce(
                    i_volumes.begin(), i_volumes.end(), 0.0F, std::plus<>(),
                    [num_neighbors_i](const auto& volume) {
                        return std::exp(volume - (1.0F / static_cast<float>(num_neighbors_i)));
                    });
            }
        });
}

unsigned int ContinuousCoordination::getNumberOfCoordinations()
{
    return m_powers.size() + static_cast<int>(m_compute_log) + static_cast<int>(m_compute_exp);
}

}; }; // end namespace freud::density
