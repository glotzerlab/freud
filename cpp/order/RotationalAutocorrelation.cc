// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "RotationalAutocorrelation.h"

#include <math.h>

using namespace std;
using namespace tbb;

/*! \file RotationalAutocorrelationFunction.cc
    \brief Implements the RotationalAutocorrelationFunction class.
*/

namespace freud { namespace order {

// This convenience function wraps exponentiation for complex numbers to avoid
// the case where pow((0,0), 0) returns (nan, nan).
std::complex<float> cpow(std::complex<float> base, float p)
    {
    if (p == 0)
        {
        return std::complex<float> (1, 0);
        }
    else
        {
        return pow(base, p);
        }
    }

// This convenience function wraps std's gamma function for factorials, with
// the appropriate shift by 1.
float factorial(int n)
    {
    return tgamma(n+1);
    }

std::pair<std::complex<float>, std::complex<float> > quat_to_greek(const quat<float> &q)
    {
    std::complex<float> xi(q.v.x, q.v.y);
    std::complex<float> zeta(q.v.z, q.s);
    return std::pair<std::complex<float>, std::complex<float> >(xi, zeta);
    }


std::complex<float> hypersphere_harmonic(const std::complex<float> xi, std::complex<float> zeta,
                                         const int l, const int m1, const int m2)
    {
    const int a = -(m1 - l/2);
    const int b = -(m2 - l/2);

    // Doing a summation over non-negative exponents, which requires the additional inner conditional.
    std::complex<float> sum_tracker(0,0);
    for (int k = 0; k <= min(a, b); k++)
        {
        if (l + k - a - b >= 0)
            {
            sum_tracker += cpow(std::conj(xi),k) * cpow(zeta, b-k) *
                              cpow(std::conj(zeta), a-k) * cpow(-xi, l+k-a-b) /
                              factorial(k) / factorial(l+k-a-b) /
                              factorial(a-k) / factorial(b-k);
            }
        }
    sum_tracker *= sqrt(factorial(a) * factorial(l-a) * factorial(b) * factorial(l-b) / (float(l)+1));
    return sum_tracker;
    }

void RotationalAutocorrelationFunction::compute(const quat<float> *ref_ors, const quat<float> *ors, unsigned int Np)
    {
    assert(ref_ors);
    assert(ors);
    assert(Np > 0);
    assert(ref_ors.size == ors.size);

    // Resize array if needed. No need to reset memory, we can do that in the loop (in parallel).
    if (Np != m_Np)
        {
        m_RA_array = std::shared_ptr< std::complex<float> >(
                  new std::complex<float>[Np],
                  std::default_delete<std::complex<float>[]>());
        m_Np = Np;
        }

    // Compute relevant values for all orientations in the system
    parallel_for(blocked_range<size_t>(0, Np),
            [=] (const blocked_range<size_t>& r)
        {
        for (size_t i = r.begin(); i != r.end(); ++i)
            {
            // Transform the orientation quaternions into Xi/Zeta coordinates;
            quat<float> q_i(ref_ors[i]);
            quat<float> q_t(ors[i]);
            quat<float> qq_0 = conj(q_i) * q_i;
            quat<float> qq_1 = conj(q_i) * q_t;
            std::pair<std::complex<float>, std::complex<float> > angle_0 = quat_to_greek(qq_0);
            std::pair<std::complex<float>, std::complex<float> > angle_1 = quat_to_greek(qq_1);

            // Loop through the valid quantum numbers.
            m_RA_array.get()[i] = std::complex<float>(0, 0);
            for (int m1 = -1*m_l/2; m1 <= m_l/2; m1++)
                {
                for (int m2 = -1*m_l/2; m2 <= m_l/2; m2++)
                    {
                    std::complex <float> combined_value = std::conj(
                            hypersphere_harmonic(angle_0.first, angle_0.second, m_l, m1, m2)
                            ) * hypersphere_harmonic(angle_1.first, angle_1.second, m_l, m1, m2);
                    m_RA_array.get()[i] += combined_value;
                    }
                }
            }
        });

    float RA_sum(0);
    for (unsigned int i = 0; i < Np; i++)
        {
        RA_sum += real(m_RA_array.get()[i]);
        }
    m_Ft = RA_sum / Np;
    };

}; }; // end namespace freud::order
