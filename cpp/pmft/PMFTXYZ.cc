// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include "PMFTXYZ.h"

using namespace std;
using namespace tbb;

/*! \internal
    \file PMFTXYZ.cc
    \brief Routines for computing 3D anisotropic potential of mean force
*/

namespace freud { namespace pmft {

PMFTXYZ::PMFTXYZ(float max_x, float max_y, float max_z, unsigned int n_bins_x, unsigned int n_bins_y, unsigned int n_bins_z, vec3<float> shiftvec)
    : PMFT(), m_max_x(max_x), m_max_y(max_y), m_max_z(max_z),
      m_n_bins_x(n_bins_x), m_n_bins_y(n_bins_y), m_n_bins_z(n_bins_z),
      m_n_faces(0), m_shiftvec(shiftvec)
    {
    if (n_bins_x < 1)
        throw invalid_argument("must be at least 1 bin in x");
    if (n_bins_y < 1)
        throw invalid_argument("must be at least 1 bin in y");
    if (n_bins_z < 1)
        throw invalid_argument("must be at least 1 bin in z");
    if (max_x < 0.0f)
        throw invalid_argument("max_x must be positive");
    if (max_y < 0.0f)
        throw invalid_argument("max_y must be positive");
    if (max_z < 0.0f)
        throw invalid_argument("max_z must be positive");

    // calculate dx, dy, dz
    m_dx = 2.0 * m_max_x / float(m_n_bins_x);
    m_dy = 2.0 * m_max_y / float(m_n_bins_y);
    m_dz = 2.0 * m_max_z / float(m_n_bins_z);

    if (m_dx > max_x)
        throw invalid_argument("max_x must be greater than dx");
    if (m_dy > max_y)
        throw invalid_argument("max_y must be greater than dy");
    if (m_dz > max_z)
        throw invalid_argument("max_z must be greater than dz");

    m_jacobian = m_dx * m_dy * m_dz;

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

    // precompute the bin center positions for z
    m_z_array = std::shared_ptr<float>(new float[m_n_bins_z], std::default_delete<float[]>());
    for (unsigned int i = 0; i < m_n_bins_z; i++)
        {
        float z = float(i) * m_dz;
        float nextz = float(i+1) * m_dz;
        m_z_array.get()[i] = -m_max_z + ((z + nextz) / 2.0);
        }
    // create and populate the pcf_array
    m_pcf_array = std::shared_ptr<float>(new float[m_n_bins_x*m_n_bins_y*m_n_bins_z], std::default_delete<float[]>());
    memset((void*)m_pcf_array.get(), 0, sizeof(float)*m_n_bins_x*m_n_bins_y*m_n_bins_z);
    m_bin_counts = std::shared_ptr<unsigned int>(new unsigned int[m_n_bins_x*m_n_bins_y*m_n_bins_z], std::default_delete<unsigned int[]>());
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_n_bins_x*m_n_bins_y*m_n_bins_z);

    m_r_cut = sqrtf(m_max_x*m_max_x + m_max_y*m_max_y + m_max_z*m_max_z);
    }

//! \internal
//! helper function to reduce the thread specific arrays into one array
void PMFTXYZ::reducePCF()
    {
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_n_bins_x*m_n_bins_y*m_n_bins_z);
    memset((void*)m_pcf_array.get(), 0, sizeof(float)*m_n_bins_x*m_n_bins_y*m_n_bins_z);
    parallel_for(blocked_range<size_t>(0,m_n_bins_x),
        [=] (const blocked_range<size_t>& r)
            {
            Index3D b_i = Index3D(m_n_bins_x, m_n_bins_y, m_n_bins_z);
            for (size_t i = r.begin(); i != r.end(); i++)
                {
                for (size_t j = 0; j < m_n_bins_y; j++)
                    {
                    for (size_t k = 0; k < m_n_bins_z; k++)
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
    float inv_jacobian = (float) 1.0 / (float) m_jacobian;
    float norm_factor = (float) 1.0 / ((float) this->m_frame_counter * (float) this->m_n_ref * (float) m_n_faces);
    // normalize pcf_array
    parallel_for(blocked_range<size_t>(0,m_n_bins_x*m_n_bins_y*m_n_bins_z),
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
void PMFTXYZ::reset()
    {
    for (tbb::enumerable_thread_specific<unsigned int *>::iterator i = m_local_bin_counts.begin(); i != m_local_bin_counts.end(); ++i)
        {
        memset((void*)(*i), 0, sizeof(unsigned int)*m_n_bins_x*m_n_bins_y*m_n_bins_z);
        }
    this->m_frame_counter = 0;
    this->m_reduce = true;
    }

//! \internal
/*! \brief Helper function to direct the calculation to the correct helper class
*/
void PMFTXYZ::accumulate(box::Box& box,
                         const locality::NeighborList *nlist,
                        vec3<float> *ref_points,
                        quat<float> *ref_orientations,
                        unsigned int n_ref,
                        vec3<float> *points,
                        quat<float> *orientations,
                        unsigned int n_p,
                        quat<float> *face_orientations,
                        unsigned int n_faces)
    {
    m_box = box;

    nlist->validate(n_ref, n_p);
    const size_t *neighbor_list(nlist->getNeighbors());

    parallel_for(blocked_range<size_t>(0,n_ref),
        [=] (const blocked_range<size_t>& r)
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

            Index3D b_i = Index3D(m_n_bins_x, m_n_bins_y, m_n_bins_z);
            Index2D q_i = Index2D(n_faces, n_p);

            bool exists;
            m_local_bin_counts.local(exists);
            if (! exists)
                {
                m_local_bin_counts.local() = new unsigned int [m_n_bins_x*m_n_bins_y*m_n_bins_z];
                memset((void*)m_local_bin_counts.local(), 0, sizeof(unsigned int)*m_n_bins_x*m_n_bins_y*m_n_bins_z);
                }

            size_t bond(nlist->find_first_index(r.begin()));

            // for each reference point
            for (size_t i = r.begin(); i != r.end(); i++)
                {
                vec3<float> ref = ref_points[i];
                // create the reference point quaternion
                quat<float> ref_q(ref_orientations[i]);

                for(; bond < nlist->getNumBonds() && neighbor_list[2*bond] == i; ++bond)
                    {
                    const size_t j(neighbor_list[2*bond + 1]);
                        {
                        // make sure that the particles are wrapped into the box
                        vec3<float> delta = m_box.wrap(points[j] - ref);
                        float rsq = dot(delta+m_shiftvec, delta+m_shiftvec);

                        // check that the particle is not checking itself
                        // 1e-6 is an arbitrary value that could be set differently if needed
                        if (rsq < 1e-6)
                            {
                            continue;
                            }
                        for (unsigned int k=0; k<n_faces; k++)
                            {
                            // create the extra quaternion
                            quat<float> qe(face_orientations[q_i(k, i)]);
                            // create point vector
                            vec3<float> v(delta);
                            // rotate the vector
                            v = rotate(conj(ref_q), v);
                            v = rotate(qe, v);

                            float x = v.x + m_max_x;
                            float y = v.y + m_max_y;
                            float z = v.z + m_max_z;

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
                            if ((ibinx < m_n_bins_x) && (ibiny < m_n_bins_y) && (ibinz < m_n_bins_z))
                                {
                                ++m_local_bin_counts.local()[b_i(ibinx, ibiny, ibinz)];
                                }
                            }
                        }
                    }
                } // done looping over reference points
            });
    this->m_frame_counter++;
    this->m_n_ref = n_ref;
    this->m_n_p = n_p;
    m_n_faces = n_faces;
    // flag to reduce
    this->m_reduce = true;
    }

}; }; // end namespace freud::pmft
