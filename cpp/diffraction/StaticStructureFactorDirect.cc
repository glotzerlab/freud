// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <random>
#include <stdexcept>
#include <tbb/concurrent_vector.h>

#include "Eigen/Eigen/Dense"

#include "Box.h"
#include "ManagedArray.h"
#include "NeighborQuery.h"
#include "StaticStructureFactorDirect.h"
#include "utils.h"

/*! \file StaticStructureFactorDirect.cc
    \brief Routines for computing static structure factors.
*/

namespace freud { namespace diffraction {

StaticStructureFactorDirect::StaticStructureFactorDirect(unsigned int bins, float k_max, float k_min, unsigned int max_k_points)
: m_max_k_points(max_k_points)
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

    // We must construct two separate histograms, one for the counts and one
    // for the actual correlation function. The counts are used to normalize
    // the correlation function.
    auto axes = StructureFactorHistogram::Axes {std::make_shared<util::RegularAxis>(bins, k_min, k_max)};
    m_structure_factor = StructureFactorHistogram(axes);
    m_local_structure_factor = StructureFactorHistogram::ThreadLocalHistogram(m_structure_factor);
    m_histogram = KBinHistogram(axes);
    m_local_histograms = KBinHistogram::ThreadLocalHistogram(m_histogram);
}

void StaticStructureFactorDirect::accumulate(const freud::locality::NeighborQuery* neighbor_query,
                                             const vec3<float>* query_points, unsigned int n_query_points,
                                             unsigned int n_total)
{
    // Compute k vectors by sampling reciprocal space
    auto const& box = neighbor_query->getBox();
    if (box.is2D()) {
        throw std::invalid_argument("2D boxes are not currently supported.");
    }
    auto const k_bin_edges = m_structure_factor.getBinEdges()[0];
    auto const k_min = k_bin_edges.front();
    auto const k_max = k_bin_edges.back();
    m_k_points = reciprocal_isotropic(box, k_max, k_min, m_max_k_points);

    // The minimum k value of validity is 4 * pi / L, where L is the smallest side length.
    // This is equal to 2 * pi / r_max.
    auto const box_L = box.getL();
    auto const min_box_length
        = box.is2D() ? std::min(box_L.x, box_L.y) : std::min(box_L.x, std::min(box_L.y, box_L.z));
    m_min_valid_k = std::min(m_min_valid_k, freud::constants::TWO_PI / min_box_length);


    // Compute F_k for the points
    auto const F_k_points = StaticStructureFactorDirect::compute_F_k(
        neighbor_query->getPoints(), neighbor_query->getNPoints(), n_total, m_k_points);

    // Compute F_k for the query points (if necessary) and compute the product S_k
    std::vector<float> S_k_all_points;
    if (query_points != nullptr)
    {
        auto const F_k_query_points = StaticStructureFactorDirect::compute_F_k(query_points, n_query_points,
                                                                               n_total, m_k_points);
        S_k_all_points = StaticStructureFactorDirect::compute_S_k(F_k_points, F_k_query_points);
    }
    else
    {
        S_k_all_points = StaticStructureFactorDirect::compute_S_k(F_k_points, F_k_points);
    }

    // Bin the S_k values and track the number of k values in each bin
    util::forLoopWrapper(0, m_k_points.size(), [&](size_t begin_k_index, size_t end_k_index) {
        for (size_t k_index = begin_k_index; k_index < end_k_index; ++k_index)
        {
            auto const k_vec = m_k_points[k_index];
            auto const k_magnitude = std::sqrt(dot(k_vec, k_vec));
            auto const k_bin = m_structure_factor.bin({k_magnitude});
            m_local_structure_factor.increment(k_bin, S_k_all_points[k_index]);
            m_local_histograms.increment(k_bin);
        };
    });
    m_reduce = true;
}

void StaticStructureFactorDirect::reduce()
{
    auto const axis_size = m_structure_factor.getAxisSizes()[0];
    m_histogram.prepare(axis_size);
    m_structure_factor.prepare(axis_size);

    // Reduce the bin counts over all threads, then use them to normalize the
    // structure factor when computing. This computes a "binned mean" over all k
    // points. Unlike some other methods in freud, no "frame counter" is needed
    // because the binned mean accounts for accumulation over frames.
    m_histogram.reduceOverThreads(m_local_histograms);
    m_structure_factor.reduceOverThreadsPerBin(
        m_local_structure_factor, [&](size_t i) { m_structure_factor[i] /= m_histogram[i]; });
}

std::vector<std::complex<float>> StaticStructureFactorDirect::compute_F_k(const vec3<float>* points,
                                                                          unsigned int n_points,
                                                                          unsigned int n_total,
                                                                          const std::vector<vec3<float>>& k_points)
{
    auto const n_k_points = k_points.size();
    auto F_k = std::vector<std::complex<float>>(n_k_points);
    std::complex<float> const normalization(1.0F / (float)std::sqrt(n_total));

    util::forLoopWrapper(0, n_k_points, [&](size_t begin, size_t end) {
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

std::vector<float>
StaticStructureFactorDirect::compute_S_k(const std::vector<std::complex<float>>& F_k_points,
                                         const std::vector<std::complex<float>>& F_k_query_points)
{
    auto const n_k_points = F_k_points.size();
    auto S_k = std::vector<float>(n_k_points);
    util::forLoopWrapper(0, n_k_points, [&](size_t begin, size_t end) {
        for (size_t k_index = begin; k_index < end; ++k_index)
        {
            S_k[k_index] = std::real(std::conj(F_k_points[k_index]) * F_k_query_points[k_index]);
        }
    });
    return S_k;
}

inline Eigen::Matrix3f box_to_matrix(const box::Box& box)
{
    // build the Eigen matrix
    Eigen::Matrix3f mat;
    for (unsigned int i = 0; i < 3; i++)
    {
        auto const box_vector = box.getLatticeVector(i);
        mat(i, 0) = box_vector.x;
        mat(i, 1) = box_vector.y;
        mat(i, 2) = box_vector.z;
    }
    return mat;
}

inline float get_prune_distance(unsigned int max_k_points, float q_max, float q_volume)
{
    if (max_k_points > M_PI * std::pow(q_max, 3.0) / (6 * q_volume))
    {
        // Above this limit, all points are used and no pruning occurs.
        return std::numeric_limits<float>::infinity();
    }
    // We use Cardano's formula to compute the pruning distance.
    auto const p = -0.75F * q_max * q_max;
    auto const q = 3.0F * static_cast<float>(max_k_points) * q_volume / static_cast<float>(M_PI)
        - q_max * q_max * q_max / 4.0F;
    auto const D = p * p * p / 27.0F + q * q / 4.0F;

    auto const u = std::pow(-std::complex<float>(q / 2.0F) + std::sqrt(std::complex<float>(D)), 1.0F / 3.0F);
    auto const v = std::pow(-std::complex<float>(q / 2.0F) - std::sqrt(std::complex<float>(D)), 1.0F / 3.0F);
    auto const x = -(u + v) / 2.0F - std::complex<float>(0.0F, 1.0F) * (u - v) * std::sqrt(3.0F) / 2.0F;
    return std::real(x) + q_max / 2.0F;
}

std::vector<vec3<float>> reciprocal_isotropic(const box::Box& box, float k_max, float k_min,
                                              unsigned int max_k_points)
{
    auto const box_matrix = box_to_matrix(box);
    // B holds "crystallographic" reciprocal box vectors that lack the factor of 2 pi.
    auto const B = box_matrix.transpose().inverse();
    auto const q_max = k_max / freud::constants::TWO_PI;
    auto const q_max_sq = q_max * q_max;
    auto const q_min = k_min / freud::constants::TWO_PI;
    auto const q_min_sq = q_min * q_min;
    auto const dq_x = B.row(0).norm();
    auto const dq_y = B.row(1).norm();
    auto const dq_z = B.row(2).norm();
    auto const q_volume = dq_x * dq_y * dq_z;

    // Above the pruning distance, the grid of k points is sampled isotropically
    // at a lower density.
    auto const q_prune_distance = get_prune_distance(max_k_points, q_max, q_volume);
    auto const q_prune_distance_sq = q_prune_distance * q_prune_distance;

    auto const bx = freud::constants::TWO_PI * vec3<float>(B(0, 0), B(0, 1), B(0, 2));
    auto const by = freud::constants::TWO_PI * vec3<float>(B(1, 0), B(1, 1), B(1, 2));
    auto const bz = freud::constants::TWO_PI * vec3<float>(B(2, 0), B(2, 1), B(2, 2));
    auto const N_kx = static_cast<unsigned int>(std::ceil(q_max / dq_x));
    auto const N_ky = static_cast<unsigned int>(std::ceil(q_max / dq_y));
    auto const N_kz = static_cast<unsigned int>(std::ceil(q_max / dq_z));

    // The maximum number of k points is a guideline. The true number of sampled
    // k points can be less or greater than max_k_points, depending on the
    // result of the random pruning procedure. Therefore, we cannot allocate a
    // fixed size for the data. Also, reserving capacity for the concurrent
    // vector had no measureable effect on performance.
    tbb::concurrent_vector<vec3<float>> k_points;

    // This is a 3D loop but we parallelize in 1D because we only need to seed
    // the random number generators once per block and there is no benefit of
    // locality if we parallelize in 2D or 3D.
    util::forLoopWrapper(0, N_kx, [&](size_t begin, size_t end) {
        // Set up thread-local random number generator for k point pruning.
        auto const thread_start = static_cast<unsigned int>(begin);
        std::random_device rd;
        std::seed_seq seed {thread_start, rd(), rd(), rd()};
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> base_dist(0, 1);
        auto random_prune = [&]() { return base_dist(rng); };

        auto const add_all_k_points = std::isinf(q_prune_distance);

        for (unsigned int kx = begin; kx < end; ++kx)
        {
            auto const k_vec_x = static_cast<float>(kx) * bx;
            for (unsigned int ky = 0; ky < N_ky; ++ky)
            {
                auto const k_vec_xy = k_vec_x + static_cast<float>(ky) * by;

                // Solve the quadratic equation for kz to limit which kz values we must sample:
                // k_min^2 <= |k_vec_xy|^2 + kz^2 |bz|^2 - 2 kz (k_vec_xy \cdot bz) <= k_max^2
                // 0 <= kz^2 (|bz|^2) + kz (-2 (k_vec_xy \cdot bz)) + (|k_vec_xy|^2 - k_min^2)
                // 0 >= kz^2 (|bz|^2) + kz (-2 (k_vec_xy \cdot bz)) + (|k_vec_xy|^2 - k_max^2)
                // This step improves performance significantly when k_min > 0
                // by eliminating a large portion of the search space. Likewise,
                // it eliminates the portion of search space outside a sphere
                // with radius k_max. We round kz_min down and kz_max up to
                // ensure that we don't accidentally throw out valid k points in
                // the range (k_min, k_max) due to rounding error.
                auto const coef_a = dot(bz, bz);
                auto const coef_b = -2.0f * dot(k_vec_xy, bz);
                auto const coef_c_min = dot(k_vec_xy, k_vec_xy) - k_min * k_min;
                auto const coef_c_max = dot(k_vec_xy, k_vec_xy) - k_max * k_max;
                auto const b_over_2a = coef_b / (2 * coef_a);
                auto const kz_min = static_cast<unsigned int>(
                    std::floor(-b_over_2a + std::sqrt(b_over_2a * b_over_2a - coef_c_min / coef_a)));
                auto const kz_max = static_cast<unsigned int>(
                    std::ceil(-b_over_2a + std::sqrt(b_over_2a * b_over_2a - coef_c_max / coef_a)));
                for (unsigned int kz = kz_min; kz < std::min(kz_max, N_kz); ++kz)
                {
                    auto const k_vec = k_vec_xy + static_cast<float>(kz) * bz;
                    auto const q_distance_sq
                        = dot(k_vec, k_vec) / freud::constants::TWO_PI / freud::constants::TWO_PI;

                    // The k vector is kept with probability min(1, (q_prune_distance / q_distance)^2).
                    // This sampling scheme aims to have a constant density of k vectors with respect to
                    // radial distance.
                    if (q_distance_sq <= q_max_sq && q_distance_sq >= q_min_sq)
                    {
                        auto const prune_probability = q_prune_distance_sq / q_distance_sq;
                        if (add_all_k_points || prune_probability > random_prune())
                        {
                            k_points.emplace_back(k_vec);
                        }
                    }
                }
            }
        }
    });
    return std::vector<vec3<float>>(k_points.cbegin(), k_points.cend());
}

}; }; // namespace freud::diffraction
