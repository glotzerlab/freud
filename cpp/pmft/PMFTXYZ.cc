// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <cassert>
#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include "Index1D.h"
#include "PMFTXYZ.h"

using namespace std;
using namespace tbb;

/*! \file PMFTXYZ.cc
    \brief Routines for computing 3D potential of mean force in XYZ coordinates
*/

namespace freud { namespace pmft {

PMFTXYZ::PMFTXYZ(float x_max, float y_max, float z_max, unsigned int n_x, unsigned int n_y, unsigned int n_z,
                 vec3<float> shiftvec)
    : PMFT(), m_x_max(x_max), m_y_max(y_max), m_z_max(z_max), m_n_x(n_x), m_n_y(n_y), m_n_z(n_z),
      m_n_faces(0), m_shiftvec(shiftvec)
{
    if (n_x < 1)
        throw invalid_argument("PMFTXYZ requires at least 1 bin in X.");
    if (n_y < 1)
        throw invalid_argument("PMFTXYZ requires at least 1 bin in Y.");
    if (n_z < 1)
        throw invalid_argument("PMFTXYZ requires at least 1 bin in Z.");
    if (x_max < 0.0f)
        throw invalid_argument("PMFTXYZ requires that x_max must be positive.");
    if (y_max < 0.0f)
        throw invalid_argument("PMFTXYZ requires that y_max must be positive.");
    if (z_max < 0.0f)
        throw invalid_argument("PMFTXYZ requires that z_max must be positive.");

    // calculate dx, dy, dz
    m_dx = 2.0 * m_x_max / float(m_n_x);
    m_dy = 2.0 * m_y_max / float(m_n_y);
    m_dz = 2.0 * m_z_max / float(m_n_z);

    if (m_dx > x_max)
        throw invalid_argument("PMFTXYZ requires that dx is less than or equal to x_max.");
    if (m_dy > y_max)
        throw invalid_argument("PMFTXYZ requires that dy is less than or equal to y_max.");
    if (m_dz > z_max)
        throw invalid_argument("PMFTXYZ requires that dz is less than or equal to z_max.");

    m_jacobian = m_dx * m_dy * m_dz;

    // precompute the bin center positions for x
    m_x_array = precomputeAxisBinCenter(m_n_x, m_dx, m_x_max);
    // precompute the bin center positions for y
    m_y_array = precomputeAxisBinCenter(m_n_y, m_dy, m_y_max);
    // precompute the bin center positions for t
    m_z_array = precomputeAxisBinCenter(m_n_z, m_dz, m_z_max);

    // create and populate the pcf_array
    m_pcf_array = util::makeEmptyArray<float>(m_n_x * m_n_y * m_n_z);
    m_bin_counts = util::makeEmptyArray<unsigned int>(m_n_x * m_n_y * m_n_z);

    // Set r_cut
    m_r_cut = sqrtf(m_x_max * m_x_max + m_y_max * m_y_max + m_z_max * m_z_max);

    m_local_bin_counts.resize(m_n_x * m_n_y * m_n_z);
}

//! \internal
//! helper function to reduce the thread specific arrays into one array
void PMFTXYZ::reducePCF()
{
    float jacobian_factor = (float) 1.0 / m_jacobian;
    reduce3D(m_n_z, m_n_x, m_n_y, [jacobian_factor](size_t i) { return jacobian_factor; });
}

//! \internal
/*! \brief Function to reset the pcf array if needed e.g. calculating between new particle types
 */
void PMFTXYZ::reset()
{
    resetGeneral(m_n_x * m_n_y * m_n_z);
}

//! \internal
/*! \brief Helper function to direct the calculation to the correct helper class
 */
void PMFTXYZ::accumulate(box::Box& box, const locality::NeighborList* nlist, vec3<float>* ref_points,
                         quat<float>* ref_orientations, unsigned int n_ref, vec3<float>* points,
                         quat<float>* orientations, unsigned int n_p, quat<float>* face_orientations,
                         unsigned int n_faces)
{
    assert(ref_points);
    assert(points);
    assert(n_ref > 0);
    assert(n_p > 0);
    assert(n_faces > 0);

    // precalc some values for faster computation within the loop
    float dx_inv = 1.0f / m_dx;
    float dy_inv = 1.0f / m_dy;
    float dz_inv = 1.0f / m_dz;

    Index3D b_i = Index3D(m_n_x, m_n_y, m_n_z);
    Index2D q_i = Index2D(n_faces, n_p);

    accumulateGeneral(box, n_ref, nlist, n_p, m_n_x * m_n_y * m_n_z, [=](size_t i, size_t j) {
        vec3<float> ref = ref_points[i];
        // create the reference point quaternion
        quat<float> ref_q(ref_orientations[i]);
        // make sure that the particles are wrapped into the box
        vec3<float> delta = m_box.wrap(points[j] - ref);
        float rsq = dot(delta + m_shiftvec, delta + m_shiftvec);

        // check that the particle is not checking itself
        // 1e-6 is an arbitrary value that could be set differently if needed
        if (rsq < 1e-6)
        {
            return;
        }
        for (unsigned int k = 0; k < n_faces; k++)
        {
            // create the extra quaternion
            quat<float> qe(face_orientations[q_i(k, i)]);
            // create point vector
            vec3<float> v(delta);
            // rotate the vector
            v = rotate(conj(ref_q), v);
            v = rotate(qe, v);

            float x = v.x + m_x_max;
            float y = v.y + m_y_max;
            float z = v.z + m_z_max;

            // bin that point
            float binx = floorf(x * dx_inv);
            float biny = floorf(y * dy_inv);
            float binz = floorf(z * dz_inv);
// fast float to int conversion with truncation
#ifdef __SSE2__
            unsigned int ibinx = _mm_cvtt_ss2si(_mm_load_ss(&binx));
            unsigned int ibiny = _mm_cvtt_ss2si(_mm_load_ss(&biny));
            unsigned int ibinz = _mm_cvtt_ss2si(_mm_load_ss(&binz));
#else
            unsigned int ibinx = (unsigned int)(binx);
            unsigned int ibiny = (unsigned int)(biny);
            unsigned int ibinz = (unsigned int)(binz);
#endif

            // increment the bin
            if ((ibinx < m_n_x) && (ibiny < m_n_y) && (ibinz < m_n_z))
            {
                ++m_local_bin_counts.local()[b_i(ibinx, ibiny, ibinz)];
            }
        }
    });
}

}; }; // end namespace freud::pmft
