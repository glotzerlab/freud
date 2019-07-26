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

inline std::complex<float> RotationalAutocorrelation::hypersphere_harmonic(const std::complex<float> xi,
                                                                           std::complex<float> zeta,
                                                                           const unsigned int l,
                                                                           const unsigned int a,
                                                                           const unsigned int b)
{
    const std::complex<float> xi_conj = std::conj(xi);
    const std::complex<float> zeta_conj = std::conj(zeta);

    // Doing a summation over non-negative exponents, which requires the additional inner conditional.
    std::complex<float> sum_tracker(0, 0);
    unsigned int bound = std::min(a, b);
    for (unsigned int k = (a + b < l ? 0 : a + b - l); k <= bound; k++)
    {
        float fact_product = m_factorials.get()[k] * m_factorials.get()[l + k - a - b]
            * m_factorials.get()[a - k] * m_factorials.get()[b - k];
        sum_tracker += cpow(xi_conj, k) * cpow(zeta, b - k) * cpow(zeta_conj, a - k)
            * cpow(-xi, l + k - a - b) / fact_product;
    }
    sum_tracker *= std::sqrt(m_factorials.get()[a] * m_factorials.get()[l - a] * m_factorials.get()[b]
                             * m_factorials.get()[l - b] / (float(l) + 1));
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
    // from having to use a more expensive process (i.e. a map).
    std::complex<float> xi = std::complex<float>(0, 0);
    std::complex<float> zeta = std::complex<float>(0, 1);
    std::vector<std::complex<float>> unit_harmonics;
    for (unsigned int a = 0; a <= m_l; a++)
    {
        for (unsigned int b = 0; b <= m_l; b++)
        {
            unit_harmonics.push_back(std::conj(hypersphere_harmonic(xi, zeta, m_l, a, b)));
        }
    }

    // Parallel loop is over orientations (technically (ref_or, or) pairs).
    tbb::parallel_for(tbb::blocked_range<size_t>(0, N), [=](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i != r.end(); ++i)
        {
            // Transform the orientation quaternions into Xi/Zeta coordinates;
            quat<float> qq_1 = conj(ref_ors[i]) * ors[i];
            std::complex<float> xi = std::complex<float>(qq_1.v.x, qq_1.v.y);
            std::complex<float> zeta = std::complex<float>(qq_1.v.z, qq_1.s);

            // Loop through the valid quantum numbers.
            m_RA_array.get()[i] = std::complex<float>(0, 0);
            unsigned int uh_index = 0;
            for (unsigned int a = 0; a <= m_l; a++)
            {
                for (unsigned int b = 0; b <= m_l; b++)
                {
                    std::complex<float> combined_value
                        = unit_harmonics[uh_index] * hypersphere_harmonic(xi, zeta, m_l, a, b);
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
