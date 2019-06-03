// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "RotationalAutocorrelation.h"

#include <math.h>

/*! \file RotationalAutocorrelation.cc
    \brief Implements the RotationalAutocorrelation class.
*/

// Avoid known stdlib clashes with Windows headers.
#if defined _WIN32
#undef min
#undef max
#endif

namespace freud { namespace order {

// This function wraps exponentiation for complex numbers to avoid
// the case where pow((0,0), 0) returns (nan, nan). Additionally, it
// serves as a micro-optimization since std::pow can be slow. We can take the
// exponent to be an unsigned int since all the sums in the hyperspherical
// harmonic below guarantee this.
inline std::complex<float> cpow(std::complex<float> base, unsigned int p)
{
    if (p == 0)
    {
        return std::complex<float>(1, 0);
    }
    else
    {
        std::complex<float> val(base);
        for (unsigned int i = 1; i < p; i++)
            val *= base;
        return val;
    }
}

// This convenience function wraps std's gamma function for factorials, with
// the appropriate shift by 1.
inline float factorial(int n)
{
    return std::tgamma(n + 1);
}

inline std::pair<std::complex<float>, std::complex<float>> quat_to_greek(const quat<float>& q)
{
    std::complex<float> xi(q.v.x, q.v.y);
    std::complex<float> zeta(q.v.z, q.s);
    return std::pair<std::complex<float>, std::complex<float>>(xi, zeta);
}

inline std::complex<float> hypersphere_harmonic(const std::complex<float> xi, std::complex<float> zeta,
                                                const int l, const int m1, const int m2)
{
    const int a = -(m1 - l / 2);
    const int b = -(m2 - l / 2);

    const std::complex<float> xi_conj = std::conj(xi);
    const std::complex<float> zeta_conj = std::conj(zeta);

    // Doing a summation over non-negative exponents, which requires the additional inner conditional.
    std::complex<float> sum_tracker(0, 0);
    for (int k = std::max(0, a + b - l); k <= std::min(a, b); k++)
    {
        sum_tracker += cpow(xi_conj, k) * cpow(zeta, b - k) * cpow(zeta_conj, a - k)
            * cpow(-xi, l + k - a - b) / factorial(k) / factorial(l + k - a - b) / factorial(a - k)
            / factorial(b - k);
    }
    sum_tracker
        *= std::sqrt(factorial(a) * factorial(l - a) * factorial(b) * factorial(l - b) / (float(l) + 1));
    return sum_tracker;
}

void RotationalAutocorrelation::compute(const quat<float>* ref_ors, const quat<float>* ors, unsigned int N)
{
    assert(ref_ors);
    assert(ors);
    assert(N > 0);
    assert(ref_ors.size == ors.size);

    // Resize array if needed. No need to reset memory, we can do that in the loop (in parallel).
    if (N != m_N)
    {
        m_RA_array = std::shared_ptr<std::complex<float>>(new std::complex<float>[N],
                                                          std::default_delete<std::complex<float>[]>());
        m_N = N;
    }

    // Precompute the hyperspherical harmonics for the unit quaternion. The
    // default quaternion constructor gives a unit quaternion. We will assume
    // the same iteration order here as in the loop below to save ourselves
    // from having to use a more expensive process (like a map).
    std::pair<std::complex<float>, std::complex<float>> angle_0 = quat_to_greek(quat<float>());
    std::vector<std::complex<float>> unit_harmonics;
    for (int m1 = -1 * m_l / 2; m1 <= m_l / 2; m1++)
    {
        for (int m2 = -1 * m_l / 2; m2 <= m_l / 2; m2++)
        {
            unit_harmonics.push_back(
                std::conj(hypersphere_harmonic(angle_0.first, angle_0.second, m_l, m1, m2)));
        }
    }

    // Parallel loop is over orientations (technically (ref_or, or) pairs).
    tbb::parallel_for(tbb::blocked_range<size_t>(0, N), [=](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i != r.end(); ++i)
        {
            // Transform the orientation quaternions into Xi/Zeta coordinates;
            quat<float> qq_1 = conj(ref_ors[i]) * ors[i];
            std::pair<std::complex<float>, std::complex<float>> angle_1 = quat_to_greek(qq_1);

            // Loop through the valid quantum numbers.
            m_RA_array.get()[i] = std::complex<float>(0, 0);
            unsigned int uh_index = 0;
            for (int m1 = -1 * m_l / 2; m1 <= m_l / 2; m1++)
            {
                for (int m2 = -1 * m_l / 2; m2 <= m_l / 2; m2++)
                {
                    std::complex<float> combined_value = unit_harmonics[uh_index]
                        * hypersphere_harmonic(angle_1.first, angle_1.second, m_l, m1, m2);
                    m_RA_array.get()[i] += combined_value;
                    uh_index += 1;
                }
            }
        }
    });

    float RA_sum(0);
    for (unsigned int i = 0; i < N; i++)
    {
        RA_sum += std::real(m_RA_array.get()[i]);
    }
    m_Ft = RA_sum / N;
};

}; }; // end namespace freud::order
