// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include "PMFTXY2D.h"

using namespace std;
using namespace tbb;

/*! \internal
    \file PMFTXY2D.cc
    \brief Routines for computing 2D anisotropic potential of mean force
*/

namespace freud { namespace pmft {

PMFTXY2D::PMFTXY2D(float max_x, float max_y, unsigned int n_bins_x, unsigned int n_bins_y)
    : PMFT(), m_max_x(max_x), m_max_y(max_y), m_n_bins_x(n_bins_x), m_n_bins_y(n_bins_y)
    {
    if (n_bins_x < 1)
        throw invalid_argument("must be at least 1 bin in x");
    if (n_bins_y < 1)
        throw invalid_argument("must be at least 1 bin in y");
    if (max_x < 0.0f)
        throw invalid_argument("max_x must be positive");
    if (max_y < 0.0f)
        throw invalid_argument("max_y must be positive");
    // calculate dx, dy
    m_dx = 2.0 * m_max_x / float(m_n_bins_x);
    m_dy = 2.0 * m_max_y / float(m_n_bins_y);

    if (m_dx > max_x)
        throw invalid_argument("max_x must be greater than dx");
    if (m_dy > max_y)
        throw invalid_argument("max_y must be greater than dy");

    m_jacobian = m_dx * m_dy;

    // precompute the bin center positions for x
    m_x_array = std::shared_ptr<float>(new float[m_n_bins_x], std::default_delete<float[]>());
    for (unsigned int i = 0; i < m_n_bins_x; i++)
        {
        float x = float(i) * m_dx;
        float nextx = float(i+1) * m_dx;
        m_x_array.get()[i] = -m_max_x + ((x + nextx) / 2.0);
        }

    // precompute the bin center positions for y
    m_y_array = std::shared_ptr<float>(new float[m_n_bins_y], std::default_delete<float[]>());
    for (unsigned int i = 0; i < m_n_bins_y; i++)
        {
        float y = float(i) * m_dy;
        float nexty = float(i+1) * m_dy;
        m_y_array.get()[i] = -m_max_y + ((y + nexty) / 2.0);
        }

    // Set r_cut
    m_r_cut = sqrtf(m_max_x*m_max_x + m_max_y*m_max_y);
    m_pcf_array = std::shared_ptr<float>(new float[m_n_bins_x * m_n_bins_y], std::default_delete<float[]>());
    memset((void*)m_pcf_array.get(), 0, sizeof(float)*m_n_bins_x*m_n_bins_y);
    m_bin_counts = std::shared_ptr<unsigned int>(new unsigned int[m_n_bins_x * m_n_bins_y], std::default_delete<unsigned int[]>());
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_n_bins_x*m_n_bins_y);
    }

//! \internal
//! helper function to reduce the thread specific arrays into one array
void PMFTXY2D::reducePCF()
    {
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_n_bins_x*m_n_bins_y);
    memset((void*)m_pcf_array.get(), 0, sizeof(float)*m_n_bins_x*m_n_bins_y);
    parallel_for(blocked_range<size_t>(0,m_n_bins_x),
        [=] (const blocked_range<size_t>& r)
            {
            Index2D b_i = Index2D(m_n_bins_x, m_n_bins_y);
            for (size_t i = r.begin(); i != r.end(); i++)
                {
                for (size_t j = 0; j < m_n_bins_y; j++)
                    {
                    for (tbb::enumerable_thread_specific<unsigned int *>::const_iterator local_bins = m_local_bin_counts.begin();
                         local_bins != m_local_bin_counts.end(); ++local_bins)
                        {
                        m_bin_counts.get()[b_i((int)i, (int)j)] += (*local_bins)[b_i((int)i, (int)j)];
                        }
                    }
                }
            });
    float inv_num_dens = this->m_box.getVolume() / (float)this->m_n_p;
    float inv_jacobian = (float) 1.0 / m_jacobian;
    float norm_factor = (float) 1.0 / ((float) this->m_frame_counter * (float) this->m_n_ref);
    // normalize pcf_array
    parallel_for(blocked_range<size_t>(0,m_n_bins_x*m_n_bins_y),
        [=] (const blocked_range<size_t>& r)
            {
            for (size_t i = r.begin(); i != r.end(); i++)
                {
                m_pcf_array.get()[i] = (float)m_bin_counts.get()[i] * norm_factor * inv_jacobian * inv_num_dens;
                }
            });
    }

//! \internal
/*! \brief Function to reset the pcf array if needed e.g. calculating between new particle types
*/

void PMFTXY2D::reset()
    {
    for (tbb::enumerable_thread_specific<unsigned int *>::iterator i = m_local_bin_counts.begin(); i != m_local_bin_counts.end(); ++i)
        {
        memset((void*)(*i), 0, sizeof(unsigned int)*m_n_bins_x*m_n_bins_y);
        }
    this->m_frame_counter = 0;
    this->m_reduce = true;
    }

//! \internal
/*! \brief Helper functionto direct the calculation to the correct helper class
*/

void PMFTXY2D::accumulate(box::Box& box,
                          const locality::NeighborList *nlist,
                         vec3<float> *ref_points,
                         float *ref_orientations,
                         unsigned int n_ref,
                         vec3<float> *points,
                         float *orientations,
                         unsigned int n_p)
    {
    this->m_box = box;

    nlist->validate(n_ref, n_p);
    const size_t *neighbor_list(nlist->getNeighbors());

    parallel_for(blocked_range<size_t>(0,n_ref),
        [=] (const blocked_range<size_t>& r)
            {
            assert(ref_points);
            assert(points);
            assert(n_ref > 0);
            assert(n_p > 0);

            // precalc some values for faster computation within the loop
            float dx_inv = 1.0f / m_dx;
            float dy_inv = 1.0f / m_dy;

            Index2D b_i = Index2D(m_n_bins_x, m_n_bins_y);

            bool exists;
            m_local_bin_counts.local(exists);
            if (! exists)
                {
                m_local_bin_counts.local() = new unsigned int [m_n_bins_x*m_n_bins_y];
                memset((void*)m_local_bin_counts.local(), 0, sizeof(unsigned int)*m_n_bins_x*m_n_bins_y);
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
                        vec3<float> delta = this->m_box.wrap(points[j] - ref);
                        float rsq = dot(delta, delta);

                        // check that the particle is not checking itself
                        // 1e-6 is an arbitrary value that could be set differently if needed
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
                        if ((ibinx < m_n_bins_x) && (ibiny < m_n_bins_y))
                            {
                            ++m_local_bin_counts.local()[b_i(ibinx, ibiny)];
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
