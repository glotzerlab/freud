// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <cmath>
#include <complex>

#include "VectorMath.h"
#include "utils.h"

/*! \file StaticStructureFactorUtil.cc
    \brief Routines for computing static structure factors.
*/

namespace freud { namespace diffraction {

void compute_F_k(const vec3<float>* points, unsigned int n_points,
        const vec3<float>* k_points, unsigned int n_k_points,
        unsigned int n_total, std::complex<float>* F_k){

    std::complex<float> const normalization(float(1.) / std::sqrt(n_total));

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
}

}; }; // namespace freud::diffraction
