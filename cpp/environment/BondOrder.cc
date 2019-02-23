// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <complex>
#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include "BondOrder.h"

using namespace std;
using namespace tbb;

/*! \file BondOrder.h
    \brief Compute the bond order diagram for the system of particles.
*/

namespace freud { namespace environment {

BondOrder::BondOrder(float rmax, float k, unsigned int n, unsigned int nbins_t, unsigned int nbins_p)
    : m_box(box::Box()), m_n_ref(0), m_n_p(0), m_nbins_t(nbins_t), m_nbins_p(nbins_p),
      m_frame_counter(0), m_reduce(true)
    {
    // sanity checks, but this is actually kinda dumb if these values are 1
    if (nbins_t < 2)
        throw invalid_argument("BondOrder requires at least 2 bins in theta.");
    if (nbins_p < 2)
        throw invalid_argument("BondOrder requires at least 2 bins in phi.");
    // calculate dt, dp
    /*
    0 < \theta < 2PI; 0 < \phi < PI
    */
    m_dt = 2.0 * M_PI / float(m_nbins_t);
    m_dp = M_PI / float(m_nbins_p);
    // this shouldn't be able to happen, but it's always better to check
    if (m_dt > 2.0 * M_PI)
        throw invalid_argument("2PI must be greater than dt");
    if (m_dp > M_PI)
        throw invalid_argument("PI must be greater than dp");

    // precompute the bin center positions for t
    m_theta_array = std::shared_ptr<float>(new float[m_nbins_t], std::default_delete<float[]>());
    for (unsigned int i = 0; i < m_nbins_t; i++)
        {
        float t = float(i) * m_dt;
        float nextt = float(i+1) * m_dt;
        m_theta_array.get()[i] = ((t + nextt) / 2.0);
        }

    // precompute the bin center positions for p
    m_phi_array = std::shared_ptr<float>(new float[m_nbins_p], std::default_delete<float[]>());
    for (unsigned int i = 0; i < m_nbins_p; i++)
        {
        float p = float(i) * m_dp;
        float nextp = float(i+1) * m_dp;
        m_phi_array.get()[i] = ((p + nextp) / 2.0);
        }

    // precompute the surface area array
    m_sa_array = std::shared_ptr<float>(new float[m_nbins_t*m_nbins_p], std::default_delete<float[]>());
    memset((void*)m_sa_array.get(), 0, sizeof(float)*m_nbins_t*m_nbins_p);
    Index2D sa_i = Index2D(m_nbins_t, m_nbins_p);
    for (unsigned int i = 0; i < m_nbins_t; i++)
        {
        for (unsigned int j = 0; j < m_nbins_p; j++)
            {
            float phi = (float)j * m_dp;
            float sa = m_dt * (cos(phi) - cos(phi + m_dp));
            m_sa_array.get()[sa_i((int)i, (int)j)] = sa;
            }
        }

    // initialize the bin counts
    m_bin_counts = std::shared_ptr<unsigned int>(new unsigned int[m_nbins_t*m_nbins_p],
            std::default_delete<unsigned int[]>());
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_nbins_t*m_nbins_p);

    // initialize the bond order array
    m_bo_array = std::shared_ptr<float>(new float[m_nbins_t*m_nbins_p], std::default_delete<float[]>());
    memset((void*)m_bin_counts.get(), 0, sizeof(float)*m_nbins_t*m_nbins_p);
    }

BondOrder::~BondOrder()
    {
    for (tbb::enumerable_thread_specific<unsigned int *>::iterator i = m_local_bin_counts.begin();
            i != m_local_bin_counts.end(); ++i)
        {
        delete[] (*i);
        }
    }

void BondOrder::reduceBondOrder()
    {
    memset((void*)m_bo_array.get(), 0, sizeof(float)*m_nbins_t*m_nbins_p);
    memset((void*)m_bin_counts.get(), 0, sizeof(unsigned int)*m_nbins_t*m_nbins_p);
    parallel_for(blocked_range<size_t>(0,m_nbins_t),
      [=] (const blocked_range<size_t>& r)
      {
      Index2D sa_i = Index2D(m_nbins_t, m_nbins_p);
      for (size_t i = r.begin(); i != r.end(); i++)
          {
          for (size_t j = 0; j < m_nbins_p; j++)
              {
              for (tbb::enumerable_thread_specific<unsigned int *>::const_iterator local_bins = m_local_bin_counts.begin();
                   local_bins != m_local_bin_counts.end(); ++local_bins)
                  {
                  m_bin_counts.get()[sa_i((int)i, (int)j)] += (*local_bins)[sa_i((int)i, (int)j)];
                  }
              m_bo_array.get()[sa_i((int)i, (int)j)] = m_bin_counts.get()[sa_i((int)i, (int)j)] / m_sa_array.get()[sa_i((int)i, (int)j)];
              }
          }
      });
    Index2D sa_i = Index2D(m_nbins_t, m_nbins_p);
    for (unsigned int i=0; i<m_nbins_t; i++)
        {
        for (unsigned int j=0; j<m_nbins_p; j++)
            {
            m_bin_counts.get()[sa_i((int)i, (int)j)] = m_bin_counts.get()[sa_i((int)i, (int)j)] / (float)m_frame_counter;
            m_bo_array.get()[sa_i((int)i, (int)j)] = m_bo_array.get()[sa_i((int)i, (int)j)] / (float)m_frame_counter;
            }
        }
    }

std::shared_ptr<float> BondOrder::getBondOrder()
    {
    if (m_reduce == true)
        {
        reduceBondOrder();
        }
    m_reduce = false;
    return m_bo_array;
    }

void BondOrder::reset()
    {
    for (tbb::enumerable_thread_specific<unsigned int *>::iterator i = m_local_bin_counts.begin(); i != m_local_bin_counts.end(); ++i)
        {
        memset((void*)(*i), 0, sizeof(unsigned int)*m_nbins_t*m_nbins_p);
        }
    // reset the frame counter
    m_frame_counter = 0;
    m_reduce = true;
    }

void BondOrder::accumulate(box::Box& box,
                           const freud::locality::NeighborList *nlist,
                           vec3<float> *ref_points,
                           quat<float> *ref_orientations,
                           unsigned int n_ref,
                           vec3<float> *points,
                           quat<float> *orientations,
                           unsigned int n_p,
                           unsigned int mode)
    {
    // transform the mode from an integer to an enumerated type (enumerated in BondOrder.h)
    BondOrderMode b_mode = static_cast<BondOrderMode>(mode);

    m_box = box;

    nlist->validate(n_ref, n_p);
    const size_t *neighbor_list(nlist->getNeighbors());

    // compute the order parameter
    parallel_for(blocked_range<size_t>(0,n_ref),
        [=] (const blocked_range<size_t>& br)
            {
            float dt_inv = 1.0f / m_dt;
            float dp_inv = 1.0f / m_dp;
            Index2D sa_i = Index2D(m_nbins_t, m_nbins_p);

            bool exists;
            m_local_bin_counts.local(exists);
            if (! exists)
                {
                m_local_bin_counts.local() = new unsigned int [m_nbins_t*m_nbins_p];
                memset((void*)m_local_bin_counts.local(), 0, sizeof(unsigned int)*m_nbins_t*m_nbins_p);
                }

            size_t bond(nlist->find_first_index(br.begin()));

            for(size_t i=br.begin(); i!=br.end(); ++i)
                {
                vec3<float> ref_pos = ref_points[i];
                quat<float> ref_q(ref_orientations[i]);

                for(; bond < nlist->getNumBonds() && neighbor_list[2*bond] == i; ++bond)
                    {
                    const size_t j(neighbor_list[2*bond + 1]);
                    //compute r between the two particles
                    vec3<float> delta = m_box.wrap(points[j] - ref_pos);

                    float rsq = dot(delta, delta);
                    if (rsq > 1e-6)
                        {
                        quat<float> q(orientations[j]);
                        vec3<float> v(delta);
                        if (b_mode == obcd)
                            {
                            // give bond directions of neighboring particles rotated by the matrix that takes the
                            // orientation of particle j to the orientation of particle i.
                            v = rotate(conj(ref_q), v);
                            v = rotate(q, v);
                            }
                        else if (b_mode == lbod)
                            {
                            // give bond directions of neighboring particles rotated into the local orientation of the
                            // central particle.
                            v = rotate(conj(ref_q), v);
                            }
                        else if (b_mode == oocd)
                            {
                            // give the directors of neighboring particles rotated into the local orientation of the
                            // central particle.
                            // pick a (random vector)
                            vec3<float> z(0,0,1);
                            // rotate that vector by the orientation of the neighboring particle
                            z = rotate(q, z);
                            // get the direction of this vector with respect to the orientation of the central particle
                            v = rotate(conj(ref_q), z);
                            }

                        // NOTE that angles are defined in the "mathematical" way, rather than how most physics
                        // textbooks do it.
                        // get theta (azimuthal angle), phi (polar angle)
                        float theta = atan2f(v.y, v.x); //-Pi..Pi

                        theta = fmod(theta, 2*M_PI);
                        if (theta < 0)
                            {
                            theta += 2*M_PI;
                            }

                        // NOTE that the below has replaced the commented out expression for phi.
                        float phi = acos(v.z / sqrt(v.x*v.x + v.y*v.y + v.z*v.z)); //0..Pi

                        // bin the point
                        float bint = floorf(theta * dt_inv);
                        float binp = floorf(phi * dp_inv);
                        // fast float to int conversion with truncation
                        #ifdef __SSE2__
                        unsigned int ibint = _mm_cvtt_ss2si(_mm_load_ss(&bint));
                        unsigned int ibinp = _mm_cvtt_ss2si(_mm_load_ss(&binp));
                        #else
                        unsigned int ibint = (unsigned int)(bint);
                        unsigned int ibinp = (unsigned int)(binp);
                        #endif

                        // increment the bin
                        if ((ibint < m_nbins_t) && (ibinp < m_nbins_p))
                        {
                            ++m_local_bin_counts.local()[sa_i(ibint, ibinp)];
                        }
                    }
                }
            }
            });


    // save the last computed number of particles
    m_n_ref = n_ref;
    m_n_p = n_p;
    m_frame_counter++;
    // flag to reduce
    m_reduce = true;
    }

}; }; // end namespace freud::environment
