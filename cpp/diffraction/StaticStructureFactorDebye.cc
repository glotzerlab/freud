// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <cmath>
#include <limits>
#include <stdexcept>

#include "NeighborQuery.h"
#include "StaticStructureFactorDebye.h"
#include "utils.h"

/*! \file StaticStructureFactorDebye.cc
    \brief Routines for computing static structure factors.
*/

namespace freud { namespace diffraction {

StaticStructureFactorDebye::StaticStructureFactorDebye(unsigned int bins, float k_max, float k_min)
    : StaticStructureFactor(bins, k_max, k_min)
{}

void StaticStructureFactorDebye::accumulate(const freud::locality::NeighborQuery* neighbor_query,
                                            const vec3<float>* query_points, unsigned int n_query_points,
                                            unsigned int n_total)
{
    const auto& box = neighbor_query->getBox();
    if (box.is2D())
    {
        throw std::invalid_argument("2D boxes are not currently supported.");
    }

    // The minimum valid k value is 4 * pi / L, where L is the smallest side length.
    const auto box_L = box.getL();
    const auto min_box_length
        = box.is2D() ? std::min(box_L.x, box_L.y) : std::min(box_L.x, std::min(box_L.y, box_L.z));
    m_min_valid_k = std::min(m_min_valid_k, 2 * freud::constants::TWO_PI / min_box_length);

    const auto* const points = neighbor_query->getPoints();
    const auto n_points = neighbor_query->getNPoints();

    std::vector<float> distances(n_points * n_query_points);
    box.computeAllDistances(points, n_points, query_points, n_query_points, distances.data());

    const auto k_bin_centers = m_structure_factor.getBinCenters()[0];

    util::forLoopWrapper(0, m_structure_factor.getAxisSizes()[0], [&](size_t begin, size_t end) {
        for (size_t k_index = begin; k_index < end; ++k_index)
        {
            const auto k = k_bin_centers[k_index];
            double S_k = 0.0;
            for (const auto& distance : distances)
            {
                S_k += util::sinc(k * distance);
            }
            S_k /= static_cast<double>(n_total);
            m_local_structure_factor.increment(k_index, S_k);
        };
    });
    m_frame_counter++;
    m_reduce = true;
}

void StaticStructureFactorDebye::reduce()
{
    m_structure_factor.prepare(m_structure_factor.getAxisSizes()[0]);
    m_structure_factor.reduceOverThreadsPerBin(m_local_structure_factor, [&](size_t i) {
        m_structure_factor[i] /= static_cast<float>(m_frame_counter);
    });
}

}; }; // namespace freud::diffraction
