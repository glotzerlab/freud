// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <cmath>
#include <cstring>
#include <complex>
#include <stdexcept>

#include "kspace.h"

using namespace std;

/*! \file kspace.cc
    \brief Analyses that compute quantities in kspace
*/

namespace freud { namespace kspace {

FTdelta::FTdelta()
    : m_NK(0),
      m_Np(0),
      m_density_Re(1),
      m_density_Im(0)
    {
    }

FTdelta::~FTdelta()
    {
    // m_S_Re and m_S_Im are std::shared_ptr which need to be passed to
    // Python and which should clean up after themselves.
    // m_K, m_r, and m_q should point to arrays managed by the calling code.
    }

void FTdelta::compute()
    {
    /* S += e**(-i * dot(K, r))
       -> S_Re += cos(dot(K, r))
       -> S_Im += - sin(dot(K, r))
    */
    unsigned int NK = m_NK;
    unsigned int Np = m_Np;
    vec3<float>* K = &m_K.front();
    vec3<float>* r = &m_r.front();
    float density_Im = m_density_Im;
    float density_Re = m_density_Re;
    m_S_Re = std::shared_ptr<float>(new float[NK], std::default_delete<float[]>());
    m_S_Im = std::shared_ptr<float>(new float[NK], std::default_delete<float[]>());
    memset((void*)m_S_Re.get(), 0, sizeof(float) * NK);
    memset((void*)m_S_Im.get(), 0, sizeof(float) * NK);
    for(unsigned int i=0; i < NK; i++)
        {
        for(unsigned int j=0; j < Np; j++)
            {
            float d = dot(K[i], r[j]); // dot product of K and r
            float CosKr, negSinKr; // real and (negative) imaginary components of exp(-i K r)
            CosKr = cos(d);
            negSinKr = sin(d);
            // S += rho * exp(-i K r)
            m_S_Re.get()[i] += CosKr * density_Re + negSinKr * density_Im;
            m_S_Im.get()[i] += CosKr * density_Im - negSinKr * density_Re;
            }
        }
    }

FTsphere::FTsphere()
    : m_radius(0.5f), m_volume(4.0f * M_PI * 0.125f / 3.0f)
    {
    }

// Calculate complex FT value of a list of uniform spheres
// Complex scattering amplitude S(K) = F(K) * f(K) for the structure
// factor F(K) and form factor f(K).
void FTsphere::compute()
    {
    unsigned int NK = m_NK;
    unsigned int Np = m_Np;
    vec3<float>* K = &m_K.front();
    vec3<float>* r = &m_r.front();
    float radius = m_radius;

    /* S += e**(-i * dot(K, r))
       -> S_Re += cos(dot(K, r))
       -> S_Im += - sin(dot(K, r))
    */
    m_S_Re = std::shared_ptr<float>(new float[NK], std::default_delete<float[]>());
    m_S_Im = std::shared_ptr<float>(new float[NK], std::default_delete<float[]>());
    memset((void*)m_S_Re.get(), 0, sizeof(float) * NK);
    memset((void*)m_S_Im.get(), 0, sizeof(float) * NK);
    for(unsigned int i=0; i < NK; i++)
        {
        for(unsigned int j=0; j < Np; j++)
            {
            // Get form factor
            // Initialize with scattering density
            float f_Im(m_density_Im);
            float f_Re(m_density_Re);

            float K2 = dot(K[i], K[i]);
            // FT evaluated at K=0 is just the scattering volume
            // f(0) = volume
            // f(K) = (4.*pi*R) / K**2 * (sinc(K*R) - cos(K*R)))
            if (K2 == 0.0f)
                {
                f_Im *= m_volume;
                f_Re *= m_volume;
                }
            else
                {
                float KR = sqrtf(K2) * radius;
                float f = 4.0f * M_PI * radius / K2 * (sinf(KR)/KR - cosf(KR));
                f_Im *= f;
                f_Re *= f;
                }

            // Get structure factor
            float CosKr, negSinKr; // real and (negative) imaginary components of exp(-i K r)
            float d = dot(K[i], r[j]); // dot product of K and r
            CosKr = cos(d);
            negSinKr = sin(d);

            // S += rho * f * exp(-i K r)
            m_S_Re.get()[i] += CosKr * f_Re + negSinKr * f_Im;
            m_S_Im.get()[i] += CosKr * f_Im - negSinKr * f_Re;
            }
        }
    }

FTpolyhedron::FTpolyhedron()
    {}

void FTpolyhedron::compute()
    {
    unsigned int NK = m_NK;
    unsigned int Np = m_Np;
    float rho_Im(m_density_Im);
    float rho_Re(m_density_Re);

    unsigned int N_facet = m_params.facet.size();

    /* S += e**(-i * dot(K, r))
       -> S_Re += cos(dot(K, r))
       -> S_Im += - sin(dot(K, r))
    */
    m_S_Re = std::shared_ptr<float>(new float[NK], std::default_delete<float[]>());
    m_S_Im = std::shared_ptr<float>(new float[NK], std::default_delete<float[]>());
    memset((void*)m_S_Re.get(), 0, sizeof(float) * NK);
    memset((void*)m_S_Im.get(), 0, sizeof(float) * NK);
    // For each K point
    for(unsigned int K_idx=0; K_idx < NK; K_idx++)
        {
        float S_Re(0.0f);
        float S_Im(0.0f);
        // For each particle
        for(unsigned int p_idx=0; p_idx < Np; p_idx++)
            {
            vec3<float> r(m_r[p_idx]);
            /* The FT of an object with orientation q at a given k-space point
             * is the same as the FT of the unrotated object at a k-space
             * point rotated the opposite way. The opposite of the rotation
             * represented by a quaternion is the conjugate of the quaternion,
             * found by inverting the sign of the imaginary components.
             */
            quat<float> q(m_q[p_idx]);
            vec3<float> K(m_K[K_idx]);
            K = rotate(conj(q), K);

            // Get form factor
            // Initialize with scattering density
            float f_Im(0.0f);
            float f_Re(0.0f);

            float K2 = dot(K,K);
            // FT evaluated at K=0 is just the scattering volume
            // f(0) = volume
            if (K2 == 0.0f)
                {
                f_Re = m_params.volume;
                f_Im = 0;
                }
            else
                {
                // Use some calculus rules to rearrange into a loop over facets.

                for(unsigned int facet_idx=0; facet_idx < N_facet; facet_idx++)
                    {
                    // Project K into plane of face
                    vec3<float> norm(m_params.norm[facet_idx]);
                    float dotKnorm(dot(K,norm));
                    vec3<float> K_proj = K - norm * dotKnorm;
                    float K_proj2 = dot(K_proj, K_proj);

                    // get polygon FT (may be accelerated in the future by converting to 2D)
                    float f2D_Im(0.0f);
                    float f2D_Re(0.0f);
                    // FT evaluated at K_proj==0 is the scattering volume (area)
                    if (K_proj2 == 0.0f)
                        {
                        f2D_Re = m_params.area[facet_idx];
                        f2D_Im = 0;
                        }
                    else
                        {
                        /* f2D = -i/k^2 * \sum_0^{Nfacets - 1) \hat(z) \cdot
                         * (l_n \times k) \exp(-ik \cdot c_n) \sinc (k \cdot l/2)
                         * Noting that -i \exp(-i x) == \sin(x) - i \cos(x),
                         * we can get the real and imaginary parts as
                         * For each element in the sum,
                         * f_n = \hat(z) \cdot (l_n \times k) \sinc (k \cdot l/2) / k^2
                         * f_Re = \sin(k \cdot c_n) * f_n
                         * f_Im = - \cos(k \cdot c_n) * f_n
                         */
                        unsigned int N_vert = m_params.facet[facet_idx].size();
                        float K2inv = 1.0f/K_proj2;
                        for(unsigned int edge_idx=0; edge_idx < N_vert; edge_idx++)
                            {
                            vec3<float> r0 = m_params.vert[m_params.facet[facet_idx][edge_idx]];
                            unsigned int next_idx = edge_idx + 1;
                            if (next_idx == N_vert) next_idx = 0;
                            vec3<float> r1 = m_params.vert[m_params.facet[facet_idx][next_idx]];
                            vec3<float> l_n = r1 - r0;
                            vec3<float> c_n = (r1 + r0)*0.5f;
                            float dotKc = dot(K_proj, c_n);
                            float dotKl = dot(K_proj, l_n);
                            vec3<float> crosslK = cross(l_n, K_proj);

                            float x = dotKl*0.5f; // argument to sinc function
                            float sinc = 1.0;
                            const float eps = 0.000001;
                            if (fabs(x) > eps) sinc = sinf(x)/x;
                            float f_n = dot(norm, crosslK) * sinc * K2inv;
                            f2D_Re -= sinf(dotKc) * f_n;
                            f2D_Im -= cosf(dotKc) * f_n;
                            } // end loop over edges
                        }

                    float d = m_params.d[facet_idx];

                    // accumulate
                    float re_exp = cosf(dotKnorm*d);
                    float im_exp = -sinf(dotKnorm*d);
                    f_Im += dotKnorm*(f2D_Re*re_exp-f2D_Im*im_exp);
                    f_Re -= dotKnorm*(f2D_Im*re_exp+f2D_Re*im_exp);
                    } // end loop over facets

                f_Re /= K2;
                f_Im /= K2;
                } // end if K != 0

            // Get structure factor
            float CosKr, negSinKr; // real and (negative) imaginary components of exp(-i K r)
            float d = dot(K, rotate(conj(q),r)); // dot product of K and r (rotated back)
            CosKr = cosf(d);
            negSinKr = sinf(d);

            // S += rho * f * exp(-i K r)
            S_Re += CosKr * f_Re + negSinKr * f_Im;
            S_Im += CosKr * f_Im - negSinKr * f_Re;
            } // end loop over particles

        m_S_Re.get()[K_idx] = S_Re * rho_Re - S_Im * rho_Im;
        m_S_Im.get()[K_idx] = S_Re * rho_Im + S_Im * rho_Re;
        } // end loop over Ks
    }

//! Helper function to build FTpolyhedron parameters
/*  \param nvert Number of vertices
    \param vert list of (x,y,z) tuples
    \param nfacet Number of facets
    \param facet_offs Offset of facet i in facet vertex list (nfacet+1)
    \param facet list of vertex indices
    \param norm list of (x,y,z) tuples of facet normal vectors
    \param d list of distance to facet
    \param area list of facet areas
    \param volume particle volume
*/
void FTpolyhedron::set_params(unsigned int nvert,
               vec3<float>* vert,
               unsigned int nfacet,
               unsigned int *facet_offs,
               unsigned int *facet,
               vec3<float>* norm,
               float *d,
               float * area,
               float volume)
    {
    poly3d_param_t params;
    params.volume = volume;
    params.vert.resize(nvert);
    for (unsigned int i=0; i < nvert; i++)
        {
        params.vert[i] = vert[i];
        }

    params.facet.resize(nfacet);
    params.norm.resize(nfacet);
    params.d.resize(nfacet);
    params.area.resize(nfacet);
    for (unsigned int i=0; i < nfacet; i++)
        {
        unsigned int facet_length = facet_offs[i+1] - facet_offs[i];

        params.facet[i].resize(facet_length);
        for (unsigned int j=0 ; j < facet_length; j++)
            {
            params.facet[i][j] = facet[facet_offs[i]+j];
            }
        params.norm[i] = norm[i];
        params.area[i] = area[i];
        params.d[i] = d[i];
        }

    m_params = params;
    }

}; }; // end namespace freud::kspace
