#include <cmath>
#include <complex>
// #include <limits>
// #include <random>
// #include <stdexcept>

// #include "Box.h"
// #include "ManagedArray.h"
// #include "NeighborQuery.h"
#include "DiffractionPattern.h"
#include "VectorMath.h"
#include "utils.h"

namespace freud { namespace diffraction {

std::vector<std::complex<float>> compute_F_k(const vec3<float>* points, unsigned int n_points,
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

std::vector<float> compute_S_k(const std::vector<std::complex<float>>& F_k_points,
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