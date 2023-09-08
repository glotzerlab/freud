#include <cmath>
#include <limits>
#include <random>
#include <tbb/concurrent_vector.h>
#include <vector>

#include "Eigen/Eigen/Dense"

#include "Box.h"
#include "StructureFactorDirect.h"
#include "utils.h"

namespace freud { namespace diffraction {

inline Eigen::Matrix3f box_to_matrix(const box::Box& box)
{
    // Build an Eigen matrix from the provided box.
    Eigen::Matrix3f mat;
    for (unsigned int i = 0; i < 3; i++)
    {
        const auto box_vector = box.getLatticeVector(i);
        mat(i, 0) = box_vector.x;
        mat(i, 1) = box_vector.y;
        mat(i, 2) = box_vector.z;
    }
    return mat;
}

inline float get_prune_distance(unsigned int num_sampled_k_points, float q_max, float q_volume)
{
    if ((num_sampled_k_points > M_PI * std::pow(q_max, 3.0) / (6 * q_volume)) || (num_sampled_k_points == 0))
    {
        // Above this limit, all points are used and no pruning occurs.
        return std::numeric_limits<float>::infinity();
    }
    // We use Cardano's formula to compute the pruning distance.
    const auto p = -0.75F * q_max * q_max;
    const auto q = 3.0F * static_cast<float>(num_sampled_k_points) * q_volume / static_cast<float>(M_PI)
        - q_max * q_max * q_max / 4.0F;
    const auto D = p * p * p / 27.0F + q * q / 4.0F;

    const auto u = std::pow(-std::complex<float>(q / 2.0F) + std::sqrt(std::complex<float>(D)), 1.0F / 3.0F);
    const auto v = std::pow(-std::complex<float>(q / 2.0F) - std::sqrt(std::complex<float>(D)), 1.0F / 3.0F);
    const auto x = -(u + v) / 2.0F - std::complex<float>(0.0F, 1.0F) * (u - v) * std::sqrt(3.0F) / 2.0F;
    return std::real(x) + q_max / 2.0F;
}

std::vector<vec3<float>> StructureFactorDirect::reciprocal_isotropic(const box::Box& box, float k_max,
                                                                     float k_min,
                                                                     unsigned int num_sampled_k_points)
{
    const auto box_matrix = box_to_matrix(box);
    // B holds "crystallographic" reciprocal box vectors that lack the factor of 2 pi.
    const auto B = box_matrix.transpose().inverse();
    const auto q_max = k_max / freud::constants::TWO_PI;
    const auto q_max_sq = q_max * q_max;
    const auto q_min = k_min / freud::constants::TWO_PI;
    const auto q_min_sq = q_min * q_min;
    const auto dq_x = B.row(0).norm();
    const auto dq_y = B.row(1).norm();
    const auto dq_z = B.row(2).norm();
    const auto q_volume = dq_x * dq_y * dq_z;

    // Above the pruning distance, the grid of k points is sampled isotropically
    // at a lower density.
    const auto q_prune_distance = get_prune_distance(num_sampled_k_points, q_max, q_volume);
    const auto q_prune_distance_sq = q_prune_distance * q_prune_distance;

    const auto bx = freud::constants::TWO_PI * vec3<float>(B(0, 0), B(0, 1), B(0, 2));
    const auto by = freud::constants::TWO_PI * vec3<float>(B(1, 0), B(1, 1), B(1, 2));
    const auto bz = freud::constants::TWO_PI * vec3<float>(B(2, 0), B(2, 1), B(2, 2));
    const auto N_kx = static_cast<unsigned int>(std::ceil(q_max / dq_x));
    const auto N_ky = static_cast<unsigned int>(std::ceil(q_max / dq_y));
    const auto N_kz = static_cast<unsigned int>(std::ceil(q_max / dq_z));

    // The maximum number of k points is a guideline. The true number of sampled
    // k points can be less or greater than num_sampled_k_points, depending on the
    // result of the random pruning procedure. Therefore, we cannot allocate a
    // fixed size for the data. Also, reserving capacity for the concurrent
    // vector had no measureable effect on performance.
    tbb::concurrent_vector<vec3<float>> k_points;

    // This is a 3D loop but we parallelize in 1D because we only need to seed
    // the random number generators once per block and there is no benefit of
    // locality if we parallelize in 2D or 3D.
    util::forLoopWrapper(0, N_kx, [&](size_t begin, size_t end) {
        // Set up thread-local random number generator for k point pruning.
        const auto thread_start = static_cast<unsigned int>(begin);
        std::random_device rd;
        std::seed_seq seed {thread_start, rd(), rd(), rd()};
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> base_dist(0, 1);
        auto random_prune = [&]() { return base_dist(rng); };

        const auto add_all_k_points = std::isinf(q_prune_distance);

        for (unsigned int kx = begin; kx < end; ++kx)
        {
            const auto k_vec_x = static_cast<float>(kx) * bx;
            for (unsigned int ky = 0; ky < N_ky; ++ky)
            {
                const auto k_vec_xy = k_vec_x + static_cast<float>(ky) * by;

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
                const auto coef_a = dot(bz, bz);
                const auto coef_b = -2 * dot(k_vec_xy, bz);
                const auto coef_c_min = dot(k_vec_xy, k_vec_xy) - k_min * k_min;
                const auto coef_c_max = dot(k_vec_xy, k_vec_xy) - k_max * k_max;
                const auto b_over_2a = coef_b / (2 * coef_a);
                const auto kz_min = static_cast<unsigned int>(
                    std::floor(-b_over_2a + std::sqrt(b_over_2a * b_over_2a - coef_c_min / coef_a)));
                const auto kz_max = static_cast<unsigned int>(
                    std::ceil(-b_over_2a + std::sqrt(b_over_2a * b_over_2a - coef_c_max / coef_a)));
                for (unsigned int kz = kz_min; kz < std::min(kz_max, N_kz); ++kz)
                {
                    const auto k_vec = k_vec_xy + static_cast<float>(kz) * bz;
                    const auto q_distance_sq
                        = dot(k_vec, k_vec) / freud::constants::TWO_PI / freud::constants::TWO_PI;

                    // The k vector is kept with probability min(1, (q_prune_distance / q_distance)^2).
                    // This sampling scheme aims to have a constant density of k vectors with respect to
                    // radial distance.
                    if (q_distance_sq <= q_max_sq && q_distance_sq >= q_min_sq)
                    {
                        const auto prune_probability = q_prune_distance_sq / q_distance_sq;
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

std::vector<std::complex<float>> StructureFactorDirect::compute_F_k(const vec3<float>* points,
                                                                    unsigned int n_points,
                                                                    unsigned int n_total,
                                                                    const std::vector<vec3<float>>& k_points)
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
StructureFactorDirect::compute_S_k(const std::vector<std::complex<float>>& F_k_points,
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
