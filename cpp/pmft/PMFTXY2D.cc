// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <cassert>
#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include "Index1D.h"
#include "PMFTXY2D.h"

using namespace std;
using namespace tbb;

/*! \file PMFTXY2D.cc
    \brief Routines for computing 2D potential of mean force in XY coordinates
*/

namespace freud { namespace pmft {

PMFTXY2D::PMFTXY2D(float x_max, float y_max, unsigned int n_x, unsigned int n_y)
    : PMFT(), m_x_max(x_max), m_y_max(y_max), m_n_x(n_x), m_n_y(n_y)
{
    if (n_x < 1)
        throw invalid_argument("PMFTXY2D requires at least 1 bin in X.");
    if (n_y < 1)
        throw invalid_argument("PMFTXY2D requires at least 1 bin in Y.");
    if (x_max < 0.0f)
        throw invalid_argument("PMFTXY2D requires that x_max must be positive.");
    if (y_max < 0.0f)
        throw invalid_argument("PMFTXY2D requires that y_max must be positive.");
    // calculate dx, dy
    m_dx = 2.0 * m_x_max / float(m_n_x);
    m_dy = 2.0 * m_y_max / float(m_n_y);

    if (m_dx > x_max)
        throw invalid_argument("PMFTXY2D requires that dx is less than or equal to x_max.");
    if (m_dy > y_max)
        throw invalid_argument("PMFTXY2D requires that dy is less than or equal to y_max.");

    m_jacobian = m_dx * m_dy;

    // precompute the bin center positions for x
    m_x_array = precomputeAxisBinCenter(m_n_x, m_dx, m_x_max);
    // precompute the bin center positions for y
    m_y_array = precomputeAxisBinCenter(m_n_y, m_dy, m_y_max);

    // create and populate the pcf_array
    m_pcf_array = util::makeEmptyArray<float>(m_n_x * m_n_y);
    m_bin_counts = util::makeEmptyArray<unsigned int>(m_n_x * m_n_y);

    // Set r_cut
    m_r_cut = sqrtf(m_x_max * m_x_max + m_y_max * m_y_max);

    m_local_bin_counts.resize(m_n_x * m_n_y);
}

//! \internal
//! helper function to reduce the thread specific arrays into one array
void PMFTXY2D::reducePCF()
{
    float jacobian_factor = (float) 1.0 / m_jacobian;
    reduce2D(m_n_x, m_n_y, [jacobian_factor](size_t i) { return jacobian_factor; });
}

//! \internal
/*! \brief Function to reset the pcf array if needed e.g. calculating between new particle types
 */

void PMFTXY2D::reset()
{
    resetGeneral(m_n_x * m_n_y);
}

//! \internal
/*! \brief Helper functionto direct the calculation to the correct helper class
 */
void PMFTXY2D::accumulate(box::Box& box, const locality::NeighborList* nlist, vec3<float>* ref_points,
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

    Index2D b_i = Index2D(m_n_x, m_n_y);

    accumulateGeneral(box, n_ref, nlist, n_p, m_n_x * m_n_y, [=](size_t i, size_t j) {
        vec3<float> ref = ref_points[i];
        vec3<float> delta = this->m_box.wrap(points[j] - ref);
        float rsq = dot(delta, delta);

        // check that the particle is not checking itself
        // 1e-6 is an arbitrary value that could be set differently if needed
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

        // find the bin to increment
        float binx = floorf(x * dx_inv);
        float biny = floorf(y * dy_inv);
// fast float to int conversion with truncation
#ifdef __SSE2__
        unsigned int ibinx = _mm_cvtt_ss2si(_mm_load_ss(&binx));
        unsigned int ibiny = _mm_cvtt_ss2si(_mm_load_ss(&biny));
#else
            unsigned int ibinx = (unsigned int)(binx);
            unsigned int ibiny = (unsigned int)(biny);
#endif

        // increment the bin
        if ((ibinx < m_n_x) && (ibiny < m_n_y))
        {
            ++m_local_bin_counts.local()[b_i(ibinx, ibiny)];
        }
    });
}

}; }; // end namespace freud::pmft
