// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <random>

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

inline float get_prune_distance(unsigned int max_k_points, float q_max, float q_volume){
    if (max_k_points > M_PI * std::pow(q_max, 3.0) / (6 * q_volume)){
        // Above this limit, all points are used and no pruning occurs.
        return std::numeric_limits<float>::infinity();
    }
    // We use Cardano's formula to compute the pruning distance.
    auto const p = -0.75f * std::pow(q_max, 2.0f);
    auto const q = 3.0f * max_k_points * q_volume / M_PI - std::pow(q_max, 3.0f) / 4.0f;
    auto const D = std::pow(p / 3.0f, 3.0f) + std::pow(q / 2.0f, 2.0f);

    auto const u = std::pow(-std::complex<float>(q / 2.0f) + std::sqrt(std::complex<float>(D)), 1.0f / 3.0f);
    auto const v = std::pow(-std::complex<float>(q / 2.0f) - std::sqrt(std::complex<float>(D)), 1.0f / 3.0f);
    auto const x = -(u + v) / 2.0f - std::complex<float>(0.0f, -1.0f) * (u - v) * std::sqrt(3.0f) / 2.0f;
    return std::real(x) + q_max / 2.0f;
}

std::vector<vec3<float>> reciprocal_isotropic(const box::Box& box, float k_max, float k_min, unsigned int max_k_points){
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

    std::vector<vec3<float>> k_points;
    auto const bx = freud::constants::TWO_PI * vec3<float>(B(0, 0), B(0, 1), B(0, 2));
    auto const by = freud::constants::TWO_PI * vec3<float>(B(1, 0), B(1, 1), B(1, 2));
    auto const bz = freud::constants::TWO_PI * vec3<float>(B(2, 0), B(2, 1), B(2, 2));
    auto const N_kx = static_cast<unsigned int>(std::ceil(q_max / dq_x));
    auto const N_ky = static_cast<unsigned int>(std::ceil(q_max / dq_y));
    auto const N_kz = static_cast<unsigned int>(std::ceil(q_max / dq_z));

    // Set up random number generator for k point pruning.
    std::random_device rd;
    std::seed_seq seed {rd(), rd(), rd()};
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> base_dist(0, 1);
    auto random_prune = [&]() { return base_dist(rng); };

    for (unsigned int kx = 0; kx < N_kx; ++kx){
        for (unsigned int ky = 0; ky < N_ky; ++ky){
            for (unsigned int kz = 0; kz < N_kz; ++kz){
                auto const k_vec = static_cast<float>(kx) * bx + static_cast<float>(ky) * by + static_cast<float>(kz) * bz;
                auto const q_distance_sq = dot(k_vec, k_vec) / freud::constants::TWO_PI / freud::constants::TWO_PI;

                // The k vector is kept with probability min(1, (q_prune_distance / q_distance)^2).
                // This sampling scheme aims to have a constant density of k vectors with respect to radial distance.
                if (q_distance_sq <= q_max_sq && q_distance_sq >= q_min_sq){
                    auto const prune_probability = q_prune_distance_sq / q_distance_sq;
                    if (prune_probability > random_prune()){
                        k_points.emplace_back(k_vec);
                    }
                }
            }
        }
    }
    return k_points;
}

}; }; // namespace freud::diffraction
