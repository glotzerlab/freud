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

IntermediateScattering::IntermediateScattering(unsigned int bins, float k_max, float k_min,
                                               unsigned int num_sampled_k_points)
    : StaticStructureFactorDirect(bins, k_max, k_min, num_sampled_k_points),
      m_k_histogram_distinct(KBinHistogram(m_structure_factor_distinct.getAxes())),
      m_local_k_histograms_distinct(KBinHistogram::ThreadLocalHistogram(m_k_histogram_distinct))
{
    if (bins == 0)
    {
        throw std::invalid_argument("IntermediateScattering requires a nonzero number of bins.");
    }
    if (k_max <= 0)
    {
        throw std::invalid_argument("IntermediateScattering requires k_max to be positive.");
    }
    if (k_min < 0)
    {
        throw std::invalid_argument("IntermediateScattering requires k_min to be non-negative.");
    }
    if (k_max <= k_min)
    {
        throw std::invalid_argument("IntermediateScattering requires that k_max must be greater than k_min.");
    }
}

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

    // record the point at t=0
    static const vec3<float>* m_r0;
    if (m_first_call)
    {
        m_r0 = neighbor_query->getPoints();
        m_first_call = false;
    }
    // Compute self-part
    const auto self_part = IntermediateScattering::compute_self(
        neighbor_query->getPoints(), m_r0, neighbor_query->getNPoints(), n_total, m_k_points);

    // Compute distinct-part
    const auto distinct_part = IntermediateScattering::compute_distinct(
        neighbor_query->getPoints(), m_r0, neighbor_query->getNPoints(), n_total, m_k_points);

    std::vector<float> S_k_self_part = IntermediateScattering::compute_S_k(self_part, self_part);
    std::vector<float> S_k_distinct_part = IntermediateScattering::compute_S_k(distinct_part, distinct_part);

    // Bin the S_k values and track the number of k values in each bin.
    util::forLoopWrapper(0, m_k_points.size(), [&](size_t begin, size_t end) {
        for (size_t k_index = begin; k_index < end; ++k_index)
        {
            const auto& k_vec = m_k_points[k_index];
            const auto k_magnitude = std::sqrt(dot(k_vec, k_vec));
            const auto k_bin1 = m_structure_factor.bin({k_magnitude});
            const auto k_bin2 = m_structure_factor_distinct.bin({k_magnitude});
            m_local_structure_factor.increment(k_bin1, S_k_self_part[k_index]);
            m_local_structure_factor_distinct.increment(k_bin2, S_k_distinct_part[k_index]);
            m_local_k_histograms.increment(k_bin1);
            m_local_k_histograms_distinct.increment(k_bin2);
        }
    });

    m_reduce = true;
}

void IntermediateScattering::reduce()
{
    const auto axis_size = m_structure_factor.getAxisSizes()[0];
    m_k_histogram.prepare(axis_size);
    m_structure_factor.prepare(axis_size);
    m_k_histogram_distinct.prepare(axis_size);
    m_structure_factor_distinct.prepare(axis_size);

    // Reduce the bin counts over all threads, then use them to normalize the
    // structure factor when computing. This computes a binned mean over all k
    // points. Unlike some other methods in freud, no frame counter is needed
    // because the binned mean accounts for accumulation over frames.
    m_k_histogram.reduceOverThreads(m_local_k_histograms);
    m_structure_factor.reduceOverThreadsPerBin(m_local_structure_factor,
                                               [&](size_t i) { m_structure_factor[i] /= m_k_histogram[i]; });
    m_k_histogram.reduceOverThreads(m_local_k_histograms_distinct);
    m_structure_factor.reduceOverThreadsPerBin(m_local_structure_factor_distinct, [&](size_t i) {
        m_structure_factor_distinct[i] /= m_k_histogram_distinct[i];
    });
}

std::vector<std::complex<float>>
IntermediateScattering::compute_self(const vec3<float>* rt, const vec3<float>* r0, unsigned int n_points,
                                     unsigned int n_total, const std::vector<vec3<float>>& k_points)
{
    //
    std::vector<vec3<float>> r_i_t0(n_points); // rt - rt0 element-wisely
    util::forLoopWrapper(0, n_points, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i)
        {
            r_i_t0[i] = rt[i] - rt[0];
        }
    });

    return IntermediateScattering::compute_F_k(r_i_t0.data(), n_points, n_total, m_k_points);
}

std::vector<std::complex<float>>
IntermediateScattering::compute_distinct(const vec3<float>* rt, const vec3<float>* r0, unsigned int n_points,
                                         unsigned int n_total, const std::vector<vec3<float>>& k_points)
{
    const auto n_rij = n_points * (n_points - 1);
    std::vector<vec3<float>> r_ij(n_rij);
    size_t i = 0;

    util::forLoopWrapper(0, n_rij, [&](size_t begin, size_t end) {
        for (size_t rt_index = begin; rt_index < end; ++rt_index)
        {
            for (size_t r0_index = 0; r0_index < n_rij; ++r0_index)
            {
                if (rt_index != r0_index)
                {
                    r_ij[i] = rt[rt_index] - r0[r0_index];
                    ++i;
                }
            }
        };
    });

    return IntermediateScattering::compute_F_k(r_ij.data(), n_rij, n_total, m_k_points);
}

}} // namespace freud::diffraction
