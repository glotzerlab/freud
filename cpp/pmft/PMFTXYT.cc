// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <cassert>
#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include "Index1D.h"
#include "PMFTXYT.h"

using namespace std;
using namespace tbb;

/*! \file PMFTXYT.cc
    \brief Routines for computing potential of mean force and torque in XYT coordinates
*/

namespace freud { namespace pmft {

PMFTXYT::PMFTXYT(float x_max, float y_max, unsigned int n_x, unsigned int n_y, unsigned int n_t)
    : PMFT(), m_x_max(x_max), m_y_max(y_max), m_t_max(2.0 * M_PI), m_n_x(n_x), m_n_y(n_y), m_n_t(n_t)
{
    if (n_x < 1)
        throw invalid_argument("PMFTXYT requires at least 1 bin in X.");
    if (n_y < 1)
        throw invalid_argument("PMFTXYT requires at least 1 bin in Y.");
    if (n_t < 1)
        throw invalid_argument("PMFTXYT requires at least 1 bin in T.");
    if (x_max < 0.0f)
        throw invalid_argument("PMFTXYT requires that x_max must be positive.");
    if (y_max < 0.0f)
        throw invalid_argument("PMFTXYT requires that y_max must be positive.");
    // calculate dx, dy, dt
    m_dx = 2.0 * m_x_max / float(m_n_x);
    m_dy = 2.0 * m_y_max / float(m_n_y);
    m_dt = m_t_max / float(m_n_t);

    if (m_dx > x_max)
        throw invalid_argument("PMFTXYT requires that dx is less than or equal to x_max.");
    if (m_dy > y_max)
        throw invalid_argument("PMFTXYT requires that dy is less than or equal to y_max.");
    if (m_dt > m_t_max)
        throw invalid_argument("PMFTXYT requires that dt is less than or equal to t_max.");

    m_jacobian = m_dx * m_dy * m_dt;

    // precompute the bin center positions for x
    m_x_array = precomputeAxisBinCenter(m_n_x, m_dx, m_x_max);
    // precompute the bin center positions for y
    m_y_array = precomputeAxisBinCenter(m_n_y, m_dy, m_y_max);
    // precompute the bin center positions for t
    m_t_array = precomputeAxisBinCenter(m_n_t, m_dt, 0);

    // create and populate the pcf_array
    m_pcf_array = util::makeEmptyArray<float>(m_n_x * m_n_y * m_n_t);
    m_bin_counts = util::makeEmptyArray<unsigned int>(m_n_x * m_n_y * m_n_t);

    // Set r_cut
    m_r_cut = sqrtf(m_x_max * m_x_max + m_y_max * m_y_max);

    m_local_bin_counts.resize(m_n_x * m_n_y * m_n_t);
}

//! \internal
//! helper function to reduce the thread specific arrays into one array
void PMFTXYT::reducePCF()
{
    float jacobian_factor = (float) 1.0 / m_jacobian;
    reduce3D(m_n_t, m_n_x, m_n_y, [jacobian_factor](size_t i) { return jacobian_factor; });
}

void PMFTXYT::reset()
{
    resetGeneral(m_n_x * m_n_y * m_n_t);
}

void PMFTXYT::accumulate(box::Box& box, const locality::NeighborList* nlist, vec3<float>* ref_points,
                         float* ref_orientations, unsigned int n_ref, vec3<float>* points,
                         float* orientations, unsigned int n_p)
{
    assert(ref_points);
    assert(points);
    assert(n_ref > 0);
    assert(n_p > 0);

    // precalc some values for faster computation within the loop
    float dx_inv = 1.0f / m_dx;
    float dy_inv = 1.0f / m_dy;
    float dt_inv = 1.0f / m_dt;

    Index3D b_i = Index3D(m_n_x, m_n_y, m_n_t);

    accumulateGeneral(box, n_ref, nlist, n_p, m_n_x * m_n_y * m_n_t, [=](size_t i, size_t j) {
        vec3<float> ref = ref_points[i];
        vec3<float> delta = m_box.wrap(points[j] - ref);

        float rsq = dot(delta, delta);
        if (rsq < 1e-6)
        {
            return;
        }
        // rotate interparticle vector
        vec2<float> myVec(delta.x, delta.y);
        rotmat2<float> myMat = rotmat2<float>::fromAngle(-ref_orientations[i]);
        vec2<float> rotVec = myMat * myVec;
        float x = rotVec.x + m_x_max;
        float y = rotVec.y + m_y_max;
        // calculate angle
        float d_theta = atan2(-delta.y, -delta.x);
        float t = orientations[j] - d_theta;
        // make sure that t is bounded between 0 and 2PI
        t = fmod(t, 2 * M_PI);
        if (t < 0)
        {
            t += 2 * M_PI;
        }
        // bin that point
        float bin_x = floorf(x * dx_inv);
        float bin_y = floorf(y * dy_inv);
        float bin_t = floorf(t * dt_inv);
// fast float to int conversion with truncation
#ifdef __SSE2__
        unsigned int ibin_x = _mm_cvtt_ss2si(_mm_load_ss(&bin_x));
        unsigned int ibin_y = _mm_cvtt_ss2si(_mm_load_ss(&bin_y));
        unsigned int ibin_t = _mm_cvtt_ss2si(_mm_load_ss(&bin_t));
#else
                unsigned int ibin_x = (unsigned int)(bin_x);
                unsigned int ibin_y = (unsigned int)(bin_y);
                unsigned int ibin_t = (unsigned int)(bin_t);
#endif
        if ((ibin_x < m_n_x) && (ibin_y < m_n_y) && (ibin_t < m_n_t))
        {
            ++m_local_bin_counts.local()[b_i(ibin_x, ibin_y, ibin_t)];
        }
    });
}
}; }; // end namespace freud::pmft
