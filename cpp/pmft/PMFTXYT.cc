// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include "PMFTXYT.h"

using namespace std;
using namespace tbb;

/*! \internal
    \file PMFTXYT.cc
    \brief Routines for computing radial density functions
*/

namespace freud { namespace pmft {

PMFTXYT::PMFTXYT(float max_x, float max_y, unsigned int n_bins_x, unsigned int n_bins_y, unsigned int n_bins_t)
    : PMFT(), m_max_x(max_x), m_max_y(max_y), m_max_t(2.0*M_PI),
      m_n_bins_x(n_bins_x), m_n_bins_y(n_bins_y), m_n_bins_t(n_bins_t)
    {
    if (n_bins_x < 1)
        throw invalid_argument("must be at least 1 bin in x");
    if (n_bins_y < 1)
        throw invalid_argument("must be at least 1 bin in y");
    if (n_bins_t < 1)
        throw invalid_argument("must be at least 1 bin in t");
    if (max_x < 0.0f)
        throw invalid_argument("max_x must be positive");
    if (max_y < 0.0f)
        throw invalid_argument("max_y must be positive");
    // calculate dx, dy, dt
    m_dx = 2.0 * m_max_x / float(m_n_bins_x);
    m_dy = 2.0 * m_max_y / float(m_n_bins_y);
    m_dt = m_max_t / float(m_n_bins_t);

    if (m_dx > max_x)
        throw invalid_argument("max_x must be greater than dx");
    if (m_dy > max_y)
        throw invalid_argument("max_y must be greater than dy");
    if (m_dt > m_max_t)
        throw invalid_argument("max_t must be greater than dt");

    m_jacobian = m_dx * m_dy * m_dt;

    // precompute the bin center positions for x
    m_x_array = std::shared_ptr<float>(new float[m_n_bins_x], std::default_delete<float[]>());
    for (unsigned int i = 0; i < m_n_bins_x; i++)
        {
        float x = float(i) * m_dx;
        float next_x = float(i+1) * m_dx;
        m_x_array.get()[i] = -m_max_x + ((x + next_x) / 2.0);
        }

    // precompute the bin center positions for y
    m_y_array = std::shared_ptr<float>(new float[m_n_bins_y], std::default_delete<float[]>());
    for (unsigned int i = 0; i < m_n_bins_y; i++)
        {
        float y = float(i) * m_dy;
        float next_y = float(i+1) * m_dy;
        m_y_array.get()[i] = -m_max_y + ((y + next_y) / 2.0);
        }

    // precompute the bin center positions for t
    m_t_array = std::shared_ptr<float>(new float[m_n_bins_t], std::default_delete<float[]>());
    for (unsigned int i = 0; i < m_n_bins_t; i++)
        {
        float t = float(i) * m_dt;
        float next_t = float(i+1) * m_dt;
        m_t_array.get()[i] = ((t + next_t) / 2.0);
        }

    // create and populate the pcf_array
    m_pcf_array = std::shared_ptr<float>(new float[m_n_bins_x*m_n_bins_y*m_n_bins_t], std::default_delete<float[]>());
    memset((void*)m_pcf_array.get(), 0, sizeof(float)*m_n_bins_x*m_n_bins_y*m_n_bins_t);
    m_bin_counts = std::shared_ptr<unsigned int>(new unsigned int[m_n_bins_x*m_n_bins_y*m_n_bins_t], std::default_delete<unsigned int[]>());
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_n_bins_x*m_n_bins_y*m_n_bins_t);

    m_r_cut = sqrtf(m_max_x*m_max_x + m_max_y*m_max_y);
    }

//! \internal
//! helper function to reduce the thread specific arrays into one array
void PMFTXYT::reducePCF()
    {
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_n_bins_x*m_n_bins_y*m_n_bins_t);
    memset((void*)m_pcf_array.get(), 0, sizeof(float)*m_n_bins_x*m_n_bins_y*m_n_bins_t);
    parallel_for(blocked_range<size_t>(0,m_n_bins_x),
        [=] (const blocked_range<size_t>& r)
            {
            Index3D b_i = Index3D(m_n_bins_x, m_n_bins_y, m_n_bins_t);
            for (size_t i = r.begin(); i != r.end(); i++)
                {
                for (size_t j = 0; j < m_n_bins_y; j++)
                    {
                    for (size_t k = 0; k < m_n_bins_t; k++)
                        {
                        for (tbb::enumerable_thread_specific<unsigned int *>::const_iterator local_bins = m_local_bin_counts.begin();
                             local_bins != m_local_bin_counts.end(); ++local_bins)
                            {
                            m_bin_counts.get()[b_i((int)i, (int)j, (int)k)] += (*local_bins)[b_i((int)i, (int)j, (int)k)];
                            }
                        }
                    }
                }
            });
    float inv_num_dens = m_box.getVolume() / (float)this->m_n_p;
    float inv_jacobian = (float) 1.0 / m_jacobian;
    float norm_factor = (float) 1.0 / ((float) this->m_frame_counter * (float) this->m_n_ref);
    // normalize pcf_array
    parallel_for(blocked_range<size_t>(0,m_n_bins_x*m_n_bins_y*m_n_bins_t),
        [=] (const blocked_range<size_t>& r)
            {
            for (size_t i = r.begin(); i != r.end(); i++)
                {
                m_pcf_array.get()[i] = (float)m_bin_counts.get()[i] * norm_factor * inv_jacobian * inv_num_dens;
                }
            });
    }

void PMFTXYT::reset()
    {
    for (tbb::enumerable_thread_specific<unsigned int *>::iterator i = m_local_bin_counts.begin(); i != m_local_bin_counts.end(); ++i)
        {
        memset((void*)(*i), 0, sizeof(unsigned int)*m_n_bins_x*m_n_bins_y*m_n_bins_t);
        }
    this->m_frame_counter = 0;
    this->m_reduce = true;
    }

void PMFTXYT::accumulate(box::Box& box,
                         const locality::NeighborList *nlist,
                         vec3<float> *ref_points,
                         float *ref_orientations,
                         unsigned int n_ref,
                         vec3<float> *points,
                         float *orientations,
                         unsigned int n_p)
    {
    m_box = box;

    nlist->validate(n_ref, n_p);
    const size_t *neighbor_list(nlist->getNeighbors());

    parallel_for(blocked_range<size_t>(0, n_ref),
        [=] (const blocked_range<size_t>& r)
            {
            assert(ref_points);
            assert(points);
            assert(n_ref > 0);
            assert(n_p > 0);

            // precalc some values for faster computation within the loop
            float dx_inv = 1.0f / m_dx;
            float dy_inv = 1.0f / m_dy;
            float dt_inv = 1.0f / m_dt;

            Index3D b_i = Index3D(m_n_bins_x, m_n_bins_y, m_n_bins_t);

            bool exists;
            m_local_bin_counts.local(exists);
            if (! exists)
                {
                m_local_bin_counts.local() = new unsigned int [m_n_bins_x*m_n_bins_y*m_n_bins_t];
                memset((void*)m_local_bin_counts.local(), 0, sizeof(unsigned int)*m_n_bins_x*m_n_bins_y*m_n_bins_t);
                }

            size_t bond(nlist->find_first_index(r.begin()));

            // for each reference point
            for (size_t i = r.begin(); i != r.end(); i++)
                {
                vec3<float> ref = ref_points[i];

                for(; bond < nlist->getNumBonds() && neighbor_list[2*bond] == i; ++bond)
                    {
                    const size_t j(neighbor_list[2*bond + 1]);
                        {
                        vec3<float> delta = m_box.wrap(points[j] - ref);

                        float rsq = dot(delta, delta);
                        if (rsq < 1e-6)
                            {
                            continue;
                            }
                        // rotate interparticle vector
                        vec2<float> myVec(delta.x, delta.y);
                        rotmat2<float> myMat = rotmat2<float>::fromAngle(-ref_orientations[i]);
                        vec2<float> rotVec = myMat * myVec;
                        float x = rotVec.x + m_max_x;
                        float y = rotVec.y + m_max_y;
                        // calculate angle
                        float d_theta = atan2(-delta.y, -delta.x);
                        float t = orientations[j] - d_theta;
                        // make sure that t is bounded between 0 and 2PI
                        t = fmod(t, 2*M_PI);
                        if (t < 0)
                            {
                            t += 2*M_PI;
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

                        if ((ibin_x < m_n_bins_x) && (ibin_y < m_n_bins_y) && (ibin_t < m_n_bins_t))
                            {
                            ++m_local_bin_counts.local()[b_i(ibin_x, ibin_y, ibin_t)];
                            }
                        }
                    }
                } // done looping over reference points
            });
    this->m_frame_counter++;
    this->m_n_ref = n_ref;
    this->m_n_p = n_p;
    // flag to reduce
    this->m_reduce = true;
    }

}; }; // end namespace freud::pmft
