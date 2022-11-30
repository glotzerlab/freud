// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "IntermediateScattering.h"
#include "Box.h"
#include "ManagedArray.h"
#include "NeighborQuery.h"
#include "utils.h"

/*! \file IntermediateScattering.cc
    \brief Routines for computing intermediate scattering function.
*/

namespace freud { namespace diffraction {

void IntermediateScattering::accumulate(const freud::locality::NeighborQuery* neighbor_query,
                                        const vec3<float>* query_points, unsigned int n_query_points,
                                        unsigned int n_total)
{
    // Compute k vectors by sampling reciprocal space.
    const auto& box = neighbor_query->getBox();
    if (box.is2D())
    {
        throw std::invalid_argument("2D boxes are not currently supported.");
    }
    const auto k_bin_edges = m_structure_factor.getBinEdges()[0];
    const auto k_min = k_bin_edges.front();
    const auto k_max = k_bin_edges.back();
    if ((!box_assigned) || (box != previous_box))
    {
        previous_box = box;
        m_k_points
            = StaticStructureFactorDirect::reciprocal_isotropic(box, k_max, k_min, m_num_sampled_k_points);
        box_assigned = true;
    }

    // The minimum valid k value is 2 * pi / L, where L is the smallest side length.
    const auto box_L = box.getL();
    const auto min_box_length
        = box.is2D() ? std::min(box_L.x, box_L.y) : std::min(box_L.x, std::min(box_L.y, box_L.z));
    m_min_valid_k = std::min(m_min_valid_k, freud::constants::TWO_PI / min_box_length);

    // Compute self-part

    if (m_first_all)
    {
        m_r0 = neighbor_query.getPoints();
        m_first_call = false;
    }
    const auto m_self_part = IntermediateScattering::compute_self(
        neighbor_query.getPoints(), m_r0, neighbor_query->getNPoints(), n_total, m_k_points)

        // Compute distinct-part
        const auto m_distinct_part
        = IntermediateScattering::compute_distinct(neighbor_query.getPoints(), m_r0,
                                                   neighbor_query->getNPoints(), n_total, m_k_points)
}

std::vector<std::complex<float>>
IntermediateScattering::compute_self(const vec3<float>* rt, const vec3<float> r0, unsigned int n_points,
                                     unsigned int n_total, const std::vector<vec3<float>>& k_points)
{
    std::vector<vec3<float>> r_i_t0(n_points); // rt - r0 element-wisely
    util::forLoopWrapper(0, n_points, [&](size_t begin, size_t end))
    {
        for (size_t i = begin; i < end; ++i)
        {
            r_i_t0[i] = rt[i] - rt[0];
        }
    }

    return IntermediateScattering::compute_F_k(r_i_t0, n_points, n_total, m_k_points);
}

std::vector<std::complex<float>>
IntermediateScattering::compute_distinct(const freud::locality::NeighborQuery* neighbor_query,
                                         const std::vector<vec3<float>>& k_points)
{
    std::vector<vec3<float>> r_ij_t0(/* TODO: Is any possible to get the number of bonds in c++*/);
    // or get the all displacement vector

    return IntermediateScattering::compute_F_k(r_ij_t0, n_points, n_total, m_k_points);
}

}} // namespace freud::diffraction
