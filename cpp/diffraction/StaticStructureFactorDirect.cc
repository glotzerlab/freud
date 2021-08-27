// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "Box.h"
#include "ManagedArray.h"
#include "NeighborQuery.h"
#include "StaticStructureFactorDirect.h"
#include "utils.h"

/*! \file StaticStructureFactorDirect.cc
    \brief Routines for computing static structure factors.
*/

namespace freud { namespace diffraction {

StaticStructureFactorDirect::StaticStructureFactorDirect(unsigned int bins, float k_max, float k_min)
{
    if (bins == 0)
    {
        throw std::invalid_argument("StaticStructureFactorDirect requires a nonzero number of bins.");
    }
    if (k_max <= 0)
    {
        throw std::invalid_argument("StaticStructureFactorDirect requires k_max to be positive.");
    }
    if (k_min < 0)
    {
        throw std::invalid_argument("StaticStructureFactorDirect requires k_min to be non-negative.");
    }
    if (k_max <= k_min)
    {
        throw std::invalid_argument("StaticStructureFactorDirect requires that k_max must be greater than k_min.");
    }

    // Construct the Histogram object that will be used to track the structure factor
    auto axes
        = StructureFactorHistogram::Axes {std::make_shared<util::RegularAxis>(bins, k_min, k_max)};
    m_histogram = StructureFactorHistogram(axes);
    m_local_histograms = StructureFactorHistogram::ThreadLocalHistogram(m_histogram);
    m_k_bin_histogram = KBinHistogram(axes);
    m_k_bin_local_histograms = KBinHistogram::ThreadLocalHistogram(m_k_bin_histogram);
    m_structure_factor.prepare(bins);
}

void StaticStructureFactorDirect::accumulate(const freud::locality::NeighborQuery* neighbor_query,
                                             const vec3<float>* query_points, unsigned int n_query_points, unsigned int n_total,
                                             const vec3<float>* k_points, unsigned int n_k_points)
{
    auto const& box = neighbor_query->getBox();

    // The minimum k value of validity is 4 * pi / L, where L is the smallest side length.
    // This is equal to 2 * pi / r_max.
    auto const box_L = box.getL();
    auto const min_box_length
        = box.is2D() ? std::min(box_L.x, box_L.y) : std::min(box_L.x, std::min(box_L.y, box_L.z));
    m_min_valid_k = 2 * freud::constants::TWO_PI / min_box_length;

    // Compute F_k for the points
    auto const F_k_points = StaticStructureFactorDirect::compute_F_k(neighbor_query->getPoints(), neighbor_query->getNPoints(), n_total, k_points, n_k_points);

    // Compute F_k for the query points (if necessary) and compute the product S_k
    std::vector<float> S_k_all_points;
    if (query_points != nullptr)
    {
        auto const F_k_query_points = StaticStructureFactorDirect::compute_F_k(query_points, n_query_points, n_total, k_points, n_k_points);
        S_k_all_points = StaticStructureFactorDirect::compute_S_k(F_k_points, F_k_query_points);
    }
    else
    {
        S_k_all_points = StaticStructureFactorDirect::compute_S_k(F_k_points, F_k_points);
    }

    // Bin the S_k values and track the number of k values in each bin
    util::forLoopWrapper(0, n_k_points, [&](size_t begin_k_index, size_t end_k_index) {
        for (size_t k_index = begin_k_index; k_index < end_k_index; ++k_index)
        {
            auto const k_vec = k_points[k_index];
            auto const k_magnitude = std::sqrt(dot(k_vec, k_vec));
            auto const k_bin = m_histogram.bin({k_magnitude});
            m_local_histograms.increment(k_bin, S_k_all_points[k_index]);
            m_k_bin_local_histograms.increment(k_bin);
        };
    });
    m_frame_counter++;
    m_reduce = true;
}

void StaticStructureFactorDirect::reduce()
{
    m_structure_factor.prepare(m_histogram.shape());
    m_local_histograms.reduceInto(m_structure_factor);
    auto k_bin_counts = util::ManagedArray<unsigned int>(m_histogram.size());
    m_k_bin_local_histograms.reduceInto(k_bin_counts);

    // Normalize by the k bin counts and frame count. This computes a "binned mean" over all k points.
    util::forLoopWrapper(0, m_structure_factor.size(), [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i)
        {
            m_structure_factor[i] /= k_bin_counts[i] * static_cast<float>(m_frame_counter);
        }
    });
}

std::vector<std::complex<float>> StaticStructureFactorDirect::compute_F_k(const vec3<float>* points, unsigned int n_points,
        unsigned int n_total, const vec3<float>* k_points, unsigned int n_k_points){

    auto F_k = std::vector<std::complex<float>>(n_k_points);
    std::complex<float> const normalization(1.0f / std::sqrt(n_total));

    util::forLoopWrapper(
        0, n_k_points, [&](size_t begin, size_t end) {
            for (size_t k_index = begin; k_index < end; ++k_index)
            {
                std::complex<float> F_ki(0);
                for (size_t r_index = 0; r_index < n_points; ++r_index)
                {
                    auto const k_vec(k_points[k_index]);
                    auto const r_vec(points[r_index]);
                    auto const alpha(dot(k_vec, r_vec));
                    F_ki += std::exp(std::complex<float>(0, alpha));
                }
                F_k[k_index] = F_ki * normalization;
            }
        });
    return F_k;
}

std::vector<float> StaticStructureFactorDirect::compute_S_k(const std::vector<std::complex<float>>& F_k_points,
        const std::vector<std::complex<float>>& F_k_query_points){
    auto const n_k_points = F_k_points.size();
    auto S_k = std::vector<float>(n_k_points);
    util::forLoopWrapper(0, n_k_points, [&](size_t begin, size_t end) {
        for (size_t k_index = begin; k_index < end; ++k_index){
            S_k[k_index] = std::real(std::conj(F_k_points[k_index]) * F_k_query_points[k_index]);
        }
    });
    return S_k;
}

}; }; // namespace freud::diffraction
