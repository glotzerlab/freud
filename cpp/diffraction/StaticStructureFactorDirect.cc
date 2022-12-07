// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <cmath>
#include <complex>
#include <limits>
#include <random>
#include <stdexcept>
#include <tbb/concurrent_vector.h>

#include "Box.h"
#include "ManagedArray.h"
#include "NeighborQuery.h"
#include "StaticStructureFactorDirect.h"
#include "utils.h"

/*! \file StaticStructureFactorDirect.cc
    \brief Routines for computing static structure factors.
*/

namespace freud { namespace diffraction {

StaticStructureFactorDirect::StaticStructureFactorDirect(unsigned int bins, float k_max, float k_min,
                                                         unsigned int num_sampled_k_points)
    : StaticStructureFactor(bins, k_max, k_min), StructureFactorDirect(bins, k_max, k_min, num_sampled_k_points), StructureFactor(bins, k_max, k_min),
      m_k_histogram(KBinHistogram(m_structure_factor.getAxes())),
      m_local_k_histograms(KBinHistogram::ThreadLocalHistogram(m_k_histogram))
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
        throw std::invalid_argument(
            "StaticStructureFactorDirect requires that k_max must be greater than k_min.");
    }
}

void StaticStructureFactorDirect::accumulate(const freud::locality::NeighborQuery* neighbor_query,
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
            = StructureFactorDirect::reciprocal_isotropic(box, k_max, k_min, m_num_sampled_k_points);
        box_assigned = true;
    }

    // The minimum valid k value is 2 * pi / L, where L is the smallest side length.
    const auto box_L = box.getL();
    const auto min_box_length
        = box.is2D() ? std::min(box_L.x, box_L.y) : std::min(box_L.x, std::min(box_L.y, box_L.z));
    m_min_valid_k = std::min(m_min_valid_k, freud::constants::TWO_PI / min_box_length);

    // Compute F_k for the points.
    const auto F_k_points = StaticStructureFactorDirect::compute_F_k(
        neighbor_query->getPoints(), neighbor_query->getNPoints(), n_total, m_k_points);

    // Compute F_k for the query points (if necessary) and compute the product S_k.
    std::vector<float> S_k_all_points;
    if (query_points != nullptr)
    {
        const auto F_k_query_points
            = StaticStructureFactorDirect::compute_F_k(query_points, n_query_points, n_total, m_k_points);
        S_k_all_points = StaticStructureFactorDirect::compute_S_k(F_k_points, F_k_query_points);
    }
    else
    {
        S_k_all_points = StaticStructureFactorDirect::compute_S_k(F_k_points, F_k_points);
    }

    // Bin the S_k values and track the number of k values in each bin.
    util::forLoopWrapper(0, m_k_points.size(), [&](size_t begin, size_t end) {
        for (size_t k_index = begin; k_index < end; ++k_index)
        {
            const auto& k_vec = m_k_points[k_index];
            const auto k_magnitude = std::sqrt(dot(k_vec, k_vec));
            const auto k_bin = m_structure_factor.bin({k_magnitude});
            m_local_structure_factor.increment(k_bin, S_k_all_points[k_index]);
            m_local_k_histograms.increment(k_bin);
        };
    });
    m_reduce = true;
}

void StaticStructureFactorDirect::reduce()
{
    const auto axis_size = m_structure_factor.getAxisSizes()[0];
    m_k_histogram.prepare(axis_size);
    m_structure_factor.prepare(axis_size);

    // Reduce the bin counts over all threads, then use them to normalize the
    // structure factor when computing. This computes a binned mean over all k
    // points. Unlike some other methods in freud, no frame counter is needed
    // because the binned mean accounts for accumulation over frames.
    m_k_histogram.reduceOverThreads(m_local_k_histograms);
    m_structure_factor.reduceOverThreadsPerBin(m_local_structure_factor,
                                               [&](size_t i) { m_structure_factor[i] /= m_k_histogram[i]; });
}

std::vector<std::complex<float>>
StaticStructureFactorDirect::compute_F_k(const vec3<float>* points, unsigned int n_points,
                                         unsigned int n_total, const std::vector<vec3<float>>& k_points)
{
    const auto n_k_points = k_points.size();
    auto F_k = std::vector<std::complex<float>>(n_k_points);
    const std::complex<float> normalization(1.0F / std::sqrt(static_cast<float>(n_total)));

    util::forLoopWrapper(0, n_k_points, [&](size_t begin, size_t end) {
        for (size_t k_index = begin; k_index < end; ++k_index)
        {
            std::complex<float> F_ki(0);
            for (size_t r_index = 0; r_index < n_points; ++r_index)
            {
                const auto& k_vec(k_points[k_index]);
                const auto& r_vec(points[r_index]);
                const auto alpha(dot(k_vec, r_vec));
                F_ki += std::exp(std::complex<float>(0, alpha));
            }
            F_k[k_index] = F_ki * normalization;
        }
    });
    return F_k;
}

std::vector<float>
StaticStructureFactorDirect::compute_S_k(const std::vector<std::complex<float>>& F_k_points,
                                         const std::vector<std::complex<float>>& F_k_query_points)
{
    const auto n_k_points = F_k_points.size();
    auto S_k = std::vector<float>(n_k_points);
    util::forLoopWrapper(0, n_k_points, [&](size_t begin, size_t end) {
        for (size_t k_index = begin; k_index < end; ++k_index)
        {
            S_k[k_index] = std::real(std::conj(F_k_points[k_index]) * F_k_query_points[k_index]);
        }
    });
    return S_k;
}

}; }; // namespace freud::diffraction
