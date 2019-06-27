// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <cassert>
#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include "Index1D.h"
#include "PMFTR12.h"

using namespace std;
using namespace tbb;

/*! \file PMFTR12.cc
    \brief Routines for computing potential of mean force and torque in R12 coordinates
*/

namespace freud { namespace pmft {

PMFTR12::PMFTR12(float r_max, unsigned int n_r, unsigned int n_t1, unsigned int n_t2)
    : PMFT(), m_r_max(r_max), m_t1_max(2.0 * M_PI), m_t2_max(2.0 * M_PI), m_n_r(n_r), m_n_t1(n_t1),
      m_n_t2(n_t2)
{
    if (n_r < 1)
        throw invalid_argument("PMFTR12 requires at least 1 bin in R.");
    if (n_t1 < 1)
        throw invalid_argument("PMFTR12 requires at least 1 bin in T1.");
    if (n_t2 < 1)
        throw invalid_argument("PMFTR12 requires at least 1 bin in T2.");
    if (r_max < 0.0f)
        throw invalid_argument("PMFTR12 requires that r_max must be positive.");
    // calculate dr, dt1, dt2
    m_dr = m_r_max / float(m_n_r);
    m_dt1 = m_t1_max / float(m_n_t1);
    m_dt2 = m_t2_max / float(m_n_t2);

    if (m_dr > r_max)
        throw invalid_argument("PMFTR12 requires that dr is less than or equal to r_max.");
    if (m_dt1 > m_t1_max)
        throw invalid_argument("PMFTR12 requires that dt1 is less than or equal to t1_max.");
    if (m_dt2 > m_t2_max)
        throw invalid_argument("PMFTR12 requires that dt2 is less than or equal to t2_max.");

    // precompute the bin center positions for r
    m_r_array = precomputeArrayGeneral(m_n_r, m_dr, [](float r, float nextr) {
        return 2.0f / 3.0f * (nextr * nextr * nextr - r * r * r) / (nextr * nextr - r * r);
    });

    // calculate the jacobian array; computed as the inverse for faster use later
    m_inv_jacobian_array
        = std::shared_ptr<float>(new float[m_n_r * m_n_t1 * m_n_t2], std::default_delete<float[]>());
    Index3D b_i = Index3D(m_n_t1, m_n_t2, m_n_r);
    for (unsigned int i = 0; i < m_n_t1; i++)
    {
        for (unsigned int j = 0; j < m_n_t2; j++)
        {
            for (unsigned int k = 0; k < m_n_r; k++)
            {
                float r = m_r_array.get()[k];
                m_inv_jacobian_array.get()[b_i((int) i, (int) j, (int) k)]
                    = (float) 1.0 / (r * m_dr * m_dt1 * m_dt2);
            }
        }
    }

    // precompute the bin center positions for T1
    m_t1_array = precomputeAxisBinCenter(m_n_t1, m_dt1, 0);

    // precompute the bin center positions for T2
    m_t2_array = precomputeAxisBinCenter(m_n_t2, m_dt2, 0);

    // create and populate the pcf_array
    m_pcf_array = util::makeEmptyArray<float>(m_n_r * m_n_t1 * m_n_t2);
    m_bin_counts = util::makeEmptyArray<unsigned int>(m_n_r * m_n_t1 * m_n_t2);

    // Set r_cut
    m_r_cut = m_r_max;

    m_local_bin_counts.resize(m_n_r * m_n_t1 * m_n_t2);
}

//! \internal
//! helper function to reduce the thread specific arrays into one array
void PMFTR12::reducePCF()
{
    reduce3D(m_n_r, m_n_t1, m_n_t2, [this](size_t i) { return m_inv_jacobian_array.get()[i]; });
}

void PMFTR12::reset()
{
    resetGeneral(m_n_r * m_n_t1 * m_n_t2);
}

void PMFTR12::accumulate(box::Box& box, const locality::NeighborList* nlist, vec3<float>* ref_points,
                         float* ref_orientations, unsigned int n_ref, vec3<float>* points,
                         float* orientations, unsigned int n_p)
{
    assert(ref_points);
    assert(points);
    assert(n_ref > 0);
    assert(n_p > 0);

    float dr_inv = 1.0f / m_dr;
    float maxrsq = m_r_max * m_r_max;
    float dt1_inv = 1.0f / m_dt1;
    float dt2_inv = 1.0f / m_dt2;

    Index3D b_i = Index3D(m_n_t1, m_n_t2, m_n_r);

    accumulateGeneral(box, n_ref, nlist, n_p, m_n_r * m_n_t1 * m_n_t2, [=](size_t i, size_t j) {
        vec3<float> ref = ref_points[i];
        vec3<float> delta = m_box.wrap(points[j] - ref);
        float rsq = dot(delta, delta);
        if (rsq < 1e-6)
        {
            return;
        }
        if (rsq < maxrsq)
        {
            float r = sqrtf(rsq);
            // calculate angles
            float d_theta1 = atan2(delta.y, delta.x);
            float d_theta2 = atan2(-delta.y, -delta.x);
            float t1 = ref_orientations[i] - d_theta1;
            float t2 = orientations[j] - d_theta2;
            // make sure that t1, t2 are bounded between 0 and 2PI
            t1 = fmod(t1, 2 * M_PI);
            if (t1 < 0)
            {
                t1 += 2 * M_PI;
            }
            t2 = fmod(t2, 2 * M_PI);
            if (t2 < 0)
            {
                t2 += 2 * M_PI;
            }
            // bin that point
            float bin_r = r * dr_inv;
            float bin_t1 = floorf(t1 * dt1_inv);
            float bin_t2 = floorf(t2 * dt2_inv);
// fast float to int conversion with truncation
#ifdef __SSE2__
            unsigned int ibin_r = _mm_cvtt_ss2si(_mm_load_ss(&bin_r));
            unsigned int ibin_t1 = _mm_cvtt_ss2si(_mm_load_ss(&bin_t1));
            unsigned int ibin_t2 = _mm_cvtt_ss2si(_mm_load_ss(&bin_t2));
#else
                unsigned int ibin_r = (unsigned int)(bin_r);
                unsigned int ibin_t1 = (unsigned int)(bin_t1);
                unsigned int ibin_t2 = (unsigned int)(bin_t2);
#endif

            if ((ibin_r < m_n_r) && (ibin_t1 < m_n_t1) && (ibin_t2 < m_n_t2))
            {
                ++m_local_bin_counts.local()[b_i(ibin_t1, ibin_t2, ibin_r)];
            }
        }
    });
}

}; }; // end namespace freud::pmft
