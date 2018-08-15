// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include "PMFTR12.h"

using namespace std;
using namespace tbb;

/*! \internal
    \file PMFTR12.cc
    \brief Routines for computing radial density functions
*/

namespace freud { namespace pmft {

PMFTR12::PMFTR12(float max_r, unsigned int nbins_r, unsigned int nbins_t1, unsigned int nbins_t2)
    : PMFT(), m_max_r(max_r), m_max_t1(2.0*M_PI), m_max_t2(2.0*M_PI),
      m_nbins_r(nbins_r), m_nbins_t1(nbins_t1), m_nbins_t2(nbins_t2)
    {
    if (nbins_r < 1)
        throw invalid_argument("must be at least 1 bin in r");
    if (nbins_t1 < 1)
        throw invalid_argument("must be at least 1 bin in T1");
    if (nbins_t2 < 1)
        throw invalid_argument("must be at least 1 bin in T2");
    if (max_r < 0.0f)
        throw invalid_argument("max_r must be positive");
    // calculate dr, dt1, dt2
    m_dr = m_max_r / float(m_nbins_r);
    m_dt1 = m_max_t1 / float(m_nbins_t1);
    m_dt2 = m_max_t2 / float(m_nbins_t2);

    if (m_dr > max_r)
        throw invalid_argument("max_r must be greater than dr");
    if (m_dt1 > m_max_t1)
        throw invalid_argument("max_t1 must be greater than dt1");
    if (m_dt2 > m_max_t2)
        throw invalid_argument("max_t2 must be greater than dt2");

    // precompute the bin center positions for r
    m_r_array = std::shared_ptr<float>(new float[m_nbins_r], std::default_delete<float[]>());
    for (unsigned int i = 0; i < m_nbins_r; i++)
        {
        float r = float(i) * m_dr;
        float nextr = float(i+1) * m_dr;
        m_r_array.get()[i] = 2.0f / 3.0f * (nextr*nextr*nextr - r*r*r) / (nextr*nextr - r*r);
        }

    // calculate the jacobian array; calc'd as the inv for faster use later
    m_inv_jacobian_array = std::shared_ptr<float>(new float[m_nbins_r*m_nbins_t1*m_nbins_t2], std::default_delete<float[]>());
    Index3D b_i = Index3D(m_nbins_t1, m_nbins_t2, m_nbins_r);
    for (unsigned int i = 0; i < m_nbins_t1; i++)
        {
        for (unsigned int j = 0; j < m_nbins_t2; j++)
            {
            for (unsigned int k = 0; k < m_nbins_r; k++)
                {
                float r = m_r_array.get()[k];
                m_inv_jacobian_array.get()[b_i((int)i, (int)j, (int)k)] = (float)1.0 / (r * m_dr * m_dt1 * m_dt2);
                }
            }
        }

    // precompute the bin center positions for T1
    m_t1_array = std::shared_ptr<float>(new float[m_nbins_t1], std::default_delete<float[]>());
    for (unsigned int i = 0; i < m_nbins_t1; i++)
        {
        float T1 = float(i) * m_dt1;
        float nextT1 = float(i+1) * m_dt1;
        m_t1_array.get()[i] = ((T1 + nextT1) / 2.0);
        }

    // precompute the bin center positions for T2
    m_t2_array = std::shared_ptr<float>(new float[m_nbins_t2], std::default_delete<float[]>());
    for (unsigned int i = 0; i < m_nbins_t2; i++)
        {
        float T2 = float(i) * m_dt2;
        float nextT2 = float(i+1) * m_dt2;
        m_t2_array.get()[i] = ((T2 + nextT2) / 2.0);
        }

    // create and populate the pcf_array
    m_pcf_array = std::shared_ptr<float>(new float[m_nbins_r*m_nbins_t1*m_nbins_t2], std::default_delete<float[]>());
    memset((void*)m_pcf_array.get(), 0, sizeof(float)*m_nbins_r*m_nbins_t1*m_nbins_t2);
    m_bin_counts = std::shared_ptr<unsigned int>(new unsigned int[m_nbins_r*m_nbins_t1*m_nbins_t2], std::default_delete<unsigned int[]>());
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_nbins_r*m_nbins_t1*m_nbins_t2);

    m_r_cut = m_max_r;
    }

//! \internal
//! helper function to reduce the thread specific arrays into one array
void PMFTR12::reducePCF()
    {
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_nbins_r*m_nbins_t1*m_nbins_t2);
    memset((void*)m_pcf_array.get(), 0, sizeof(float)*m_nbins_r*m_nbins_t1*m_nbins_t2);
    parallel_for(blocked_range<size_t>(0,m_nbins_t1),
        [=] (const blocked_range<size_t>& r)
            {
            Index3D b_i = Index3D(m_nbins_t1, m_nbins_t2, m_nbins_r);
            for (size_t i = r.begin(); i != r.end(); i++)
                {
                for (size_t j = 0; j < m_nbins_t2; j++)
                    {
                    for (size_t k = 0; k < m_nbins_r; k++)
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
    float norm_factor = (float) 1.0 / ((float) this->m_frame_counter * (float) this->m_n_ref);
    // normalize pcf_array
    // avoid need to unravel b/c arrays are in the same index order
    parallel_for(blocked_range<size_t>(0,m_nbins_r*m_nbins_t1*m_nbins_t2),
        [=] (const blocked_range<size_t>& r)
            {
            for (size_t i = r.begin(); i != r.end(); i++)
                {
                m_pcf_array.get()[i] = (float)m_bin_counts.get()[i] * norm_factor * m_inv_jacobian_array.get()[i] * inv_num_dens;
                }
            });
    }

void PMFTR12::reset()
    {
    for (tbb::enumerable_thread_specific<unsigned int *>::iterator i = m_local_bin_counts.begin(); i != m_local_bin_counts.end(); ++i)
        {
        memset((void*)(*i), 0, sizeof(unsigned int)*m_nbins_r*m_nbins_t1*m_nbins_t2);
        }
    this->m_frame_counter = 0;
    this->m_reduce = true;
    }

void PMFTR12::accumulate(box::Box& box,
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
        [=] (const blocked_range<size_t>& br)
            {
            assert(ref_points);
            assert(points);
            assert(n_ref > 0);
            assert(n_p > 0);

            float dr_inv = 1.0f / m_dr;
            float maxrsq = m_max_r * m_max_r;
            float dt1_inv = 1.0f / m_dt1;
            float dt2_inv = 1.0f / m_dt2;

            Index3D b_i = Index3D(m_nbins_t1, m_nbins_t2, m_nbins_r);

            bool exists;
            m_local_bin_counts.local(exists);
            if (! exists)
                {
                m_local_bin_counts.local() = new unsigned int [m_nbins_r*m_nbins_t1*m_nbins_t2];
                memset((void*)m_local_bin_counts.local(), 0, sizeof(unsigned int)*m_nbins_r*m_nbins_t1*m_nbins_t2);
                }

            size_t bond(nlist->find_first_index(br.begin()));

            // for each reference point
            for (size_t i = br.begin(); i != br.end(); i++)
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
                        if (rsq < maxrsq)
                            {
                            float r = sqrtf(rsq);
                            // calculate angles
                            float d_theta1 = atan2(delta.y, delta.x);
                            float d_theta2 = atan2(-delta.y, -delta.x);
                            float t1 = ref_orientations[i] - d_theta1;
                            float t2 = orientations[j] - d_theta2;
                            // make sure that t1, t2 are bounded between 0 and 2PI
                            t1 = fmod(t1, 2*M_PI);
                            if (t1 < 0)
                                {
                                t1 += 2*M_PI;
                                }
                            t2 = fmod(t2, 2*M_PI);
                            if (t2 < 0)
                                {
                                t2 += 2*M_PI;
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

                            if ((ibin_r < m_nbins_r) && (ibin_t1 < m_nbins_t1) && (ibin_t2 < m_nbins_t2))
                                {
                                ++m_local_bin_counts.local()[b_i(ibin_t1, ibin_t2, ibin_r)];
                                }
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
