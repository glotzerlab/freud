// Copyright (c) 2010-2024 The Regents of the University of Michigan
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

ContinuousCoordination::ContinuousCoordination(const std::vector<float>& powers, bool compute_log, bool compute_exp)
    : m_powers(powers), m_compute_exp(compute_exp), m_compute_log(compute_log)
{
}

void ContinuousCoordination::compute(const freud::locality::Voronoi* voronoi, const freud::locality::NeighborList* nlist, bool is2D)
    {
    size_t num_points = nlist->getNumQueryPoints();
    m_coordination.prepare({num_points, getNumberOfCoordinations()});
    const auto& volumes = voronoi->getVolumes();
    const auto& num_neighbors = nlist->getCounts();
    // 2 for triangles 3 for pyramids
    const float volume_prefactor = is2D ? 2.0 : 3.0;
    freud::locality::loopOverNeighborListIterator(nlist,
        [&](size_t i, const std::shared_ptr<freud::locality::NeighborPerPointIterator>& ppiter) {
            // 1/2 comes from the distance vector since we want to measure from the pyramid
            // base to the center.
            float prefactor = 1.0 / (volume_prefactor * 2.0 * volumes[i]);
            std::vector<float> i_volumes;
            for (freud::locality::NeighborBond nb = ppiter->next(); !ppiter->end(); nb = ppiter->next()) {
                i_volumes.emplace_back(prefactor * nb.getWeight() * nb.getDistance());
                }
            size_t j{0};
            float num_neighbors_i{static_cast<float>(num_neighbors[i])};
            for (size_t k{0}; k < m_powers.size(); ++k) {
                auto cn = std::transform_reduce(
                    i_volumes.begin(),
                    i_volumes.end(),
                    0.0,
                    std::plus<float>(),
                    [*this, k](auto& volume) { return std::pow(volume, this->m_powers[k]); }
                );
                m_coordination(i, j++) = std::pow(num_neighbors_i, 2.0 - m_powers[k]) / cn;
            }
            if (m_compute_log) {
                auto cn = std::transform_reduce(
                    i_volumes.begin(),
                    i_volumes.end(),
                    0.0,
                    std::plus<float>(),
                    logf
                );
                m_coordination(i, j++) = -cn / std::log(num_neighbors_i);
            }
            if (m_compute_exp) {
                m_coordination(i, j) = std::transform_reduce(
                    i_volumes.begin(),
                    i_volumes.end(),
                    0.0,
                    std::plus<float>(),
                    [num_neighbors_i](auto& volume) { return std::exp(volume - (1.0 / static_cast<float>(num_neighbors_i))); } 
                );
            }
        });
}


unsigned int ContinuousCoordination::getNumberOfCoordinations() {
    return m_powers.size() + static_cast<int>(m_compute_log) + static_cast<int>(m_compute_exp);
}

}; }; // end namespace freud::density
