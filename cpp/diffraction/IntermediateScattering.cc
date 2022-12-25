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

IntermediateScattering::IntermediateScattering(const box::Box& box, unsigned int bins, float k_max,
                                               float k_min, unsigned int num_sampled_k_points)
    : StructureFactorDirect(bins, k_max, k_min, num_sampled_k_points), StructureFactor(bins, k_max, k_min),
      m_box(box)
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

    m_k_points = IntermediateScattering::reciprocal_isotropic(box, k_max, k_min, m_num_sampled_k_points);
}

void IntermediateScattering::compute(const vec3<float>* points, unsigned int num_points,
                                     const vec3<float>* query_points, unsigned int num_query_points,
                                     unsigned int num_frames, unsigned int n_total)
{
    // initialize the histograms, now that the size of both axes are known
    m_self_function = StructureFactorHistogram({
        std::make_shared<util::RegularAxis>(num_frames, -0.5, num_frames - 0.5),
        std::make_shared<util::RegularAxis>(m_nbins, m_k_min, m_k_max),
    });
    m_local_self_function = StructureFactorHistogram::ThreadLocalHistogram(m_self_function);
    m_distinct_function = StructureFactorHistogram({
        std::make_shared<util::RegularAxis>(num_frames, -0.5, num_frames - 0.5),
        std::make_shared<util::RegularAxis>(m_nbins, m_k_min, m_k_max),
    });
    m_local_distinct_function = StructureFactorHistogram::ThreadLocalHistogram(m_distinct_function);
    // Compute k vectors by sampling reciprocal space.
    if (m_box.is2D())
    {
        throw std::invalid_argument("2D boxes are not currently supported.");
    }

    // The minimum valid k value is 2 * pi / L, where L is the smallest side length.
    const auto box_L = m_box.getL();
    const auto min_box_length = std::min(box_L.x, std::min(box_L.y, box_L.z));
    m_min_valid_k = freud::constants::TWO_PI / min_box_length;

    // record the point at t=0
    const vec3<float>* query_r0 = &query_points[0];
    const vec3<float>* r0 = &points[0];

    util::forLoopWrapper(0, num_frames, [&](size_t begin, size_t end) {
        for (size_t t = begin; t < end; t++)
        {
            size_t offset = t * num_points;
            // Compute self-part
            const auto self_part
                = IntermediateScattering::compute_self(&points[offset], r0, num_points, n_total, m_k_points);

            // Compute distinct-part
            const auto distinct_part = IntermediateScattering::compute_distinct(
                &points[offset], query_r0, num_query_points, n_total, m_k_points);

            std::vector<float> S_k_self_part = StaticStructureFactorDirect::compute_S_k(self_part, self_part);
            std::vector<float> S_k_distinct_part
                = StaticStructureFactorDirect::compute_S_k(distinct_part, distinct_part);

            // Bin the S_k values and track the number of k values in each bin.
            util::forLoopWrapper(0, m_k_points.size(), [&](size_t begin, size_t end) {
                for (size_t k_index = begin; k_index < end; ++k_index)
                {
                    const auto& k_vec = m_k_points[k_index];
                    const auto k_magnitude = std::sqrt(dot(k_vec, k_vec));
                    const auto k_bin = m_self_function.bin({t, k_magnitude});
                    m_local_self_function.increment(k_bin, S_k_self_part[k_index]);
                    m_local_distinct_function.increment(k_bin, S_k_distinct_part[k_index]);
                    m_local_k_histograms.increment(k_bin);
                }
            });
        }
    });

    m_reduce = true;
}

void IntermediateScattering::reduce()
{
    const auto k_axis_size = m_k_histogram.getAxisSizes();
    m_k_histogram.prepare(k_axis_size);
    const auto self_axis_size = m_self_function.getAxisSizes();
    m_self_function.prepare(self_axis_size);
    const auto distinct_axis_size = m_distinct_function.getAxisSizes();
    m_distinct_function.prepare(distinct_axis_size);

    // Reduce the bin counts over all threads, then use them to normalize the
    // structure factor when computing. This computes a binned mean over all k
    // points. Unlike some other methods in freud, no frame counter is needed
    // because the binned mean accounts for accumulation over frames.
    m_k_histogram.reduceOverThreads(m_local_k_histograms);
    m_self_function.reduceOverThreadsPerBin(m_local_self_function,
                                            [&](size_t i) { m_self_function[i] /= m_k_histogram[i]; });
    m_distinct_function.reduceOverThreadsPerBin(
        m_local_distinct_function, [&](size_t i) { m_distinct_function[i] /= m_k_histogram[i]; });
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

    return StructureFactorDirect::compute_F_k(r_i_t0.data(), n_points, n_total, m_k_points);
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

    return StructureFactorDirect::compute_F_k(r_ij.data(), n_rij, n_total, m_k_points);
}

}} // namespace freud::diffraction
