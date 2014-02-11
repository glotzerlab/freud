#include "kspace.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#include <cmath>
#include <complex>

using namespace std;
using namespace boost::python;

namespace freud { namespace kspace {

FTdelta::FTdelta()
    : m_NK(0),
      m_Np(0),
      m_density_Im(0),
      m_density_Re(1)
    {
    }

FTdelta::~FTdelta()
    {
    // S_Re and S_Im are boost::shared_array which need to be passed to Python and which should clean up after themselves.
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
    float3* K = m_K;
    float3* r = m_r;
    float4* q = m_q;
    float density_Im = m_density_Im;
    float density_Re = m_density_Re;
    m_S_Re = boost::shared_array<float>(new float[NK]);
    m_S_Im = boost::shared_array<float>(new float[NK]);
    memset((void*)m_S_Re.get(), 0, sizeof(float) * NK);
    memset((void*)m_S_Im.get(), 0, sizeof(float) * NK);
    for(unsigned int i=0; i < NK; i++)
        {
        for(unsigned int j=0; j < Np; j++)
            {
            float d; // dot product of K and r
            d = K[i].x * r[j].x + K[i].y * r[j].y + K[i].z * r[j].z;
            float CosKr, negSinKr; // real and (negative) imaginary components of exp(-i K r)
            CosKr = cos(d);
            negSinKr = sin(d);
            // S += rho * exp(-i K r)
            m_S_Re[i] += CosKr * density_Re + negSinKr * density_Im;
            m_S_Im[i] += CosKr * density_Im - negSinKr * density_Re;
            }
        }
    }

void FTdelta::computePy()
    {
    // compute with the GIL released
    util::ScopedGILRelease gil;
    compute();
    }

FTsphere::FTsphere()
    : m_radius(0.5f), m_volume(4.0f * M_PI * 0.125f / 3.0f)
    {
    }

// Calculate complex FT value of a list of uniform spheres
// Complex scattering amplitude S(K) = F(K) * f(K) for the structure factor F(K) and form factor f(K).
void FTsphere::compute()
    {
    unsigned int NK = m_NK;
    unsigned int Np = m_Np;
    float3* K = m_K;
    float3* r = m_r;
    float4* q = m_q;
    float radius = m_radius;

    /* S += e**(-i * dot(K, r))
       -> S_Re += cos(dot(K, r))
       -> S_Im += - sin(dot(K, r))
    */
    m_S_Re = boost::shared_array<float>(new float[NK]);
    m_S_Im = boost::shared_array<float>(new float[NK]);
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

            float K2 = K[i].x * K[i].x + K[i].y * K[i].y + K[i].z * K[i].z;
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
            float d; // dot product of K and r
            d = K[i].x * r[j].x + K[i].y * r[j].y + K[i].z * r[j].z;
            CosKr = cos(d);
            negSinKr = sin(d);

            // S += rho * f * exp(-i K r)
            m_S_Re[i] += CosKr * f_Re + negSinKr * f_Im;
            m_S_Im[i] += CosKr * f_Im - negSinKr * f_Re;
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

    //float3* K_array = m_K;
    //float3* r = m_r;
    //float4* q = m_q;

    unsigned int N_facet = m_params.facet.size();
    unsigned int N_vert = m_params.vert.size();

    /* S += e**(-i * dot(K, r))
       -> S_Re += cos(dot(K, r))
       -> S_Im += - sin(dot(K, r))
    */
    m_S_Re = boost::shared_array<float>(new float[NK]);
    m_S_Im = boost::shared_array<float>(new float[NK]);
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
            vec3<float> r(m_r[p_idx].x, m_r[p_idx].y, m_r[p_idx].z);
            /* The FT of an object with orientation q at a given k-space point is the same as the FT
               of the unrotated object at a k-space point rotated the opposite way.
               The opposite of the rotation represented by a quaternion is the conjugate of the quaternion,
               found by inverting the sign of the imaginary components.
            */
            quat<float> q(m_q[p_idx].w, vec3<float>(m_q[p_idx].x, m_q[p_idx].y, m_q[p_idx].z));
            vec3<float> K(m_K[K_idx].x, m_K[K_idx].y, m_K[K_idx].z);
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
                f_Re = f_Im = m_params.volume;
                }
            else
                {
                // Use some calculus rules to rearrange into a loop over facets.

                for(unsigned int facet_idx=0; facet_idx < N_facet; facet_idx++)
                    {
                    // Project K into plane of face
                    vec3<float> norm(m_params.norm[p_idx]);
                    float dotKnorm(dot(K,norm));
                    vec3<float> K_proj = K - norm * dotKnorm;
                    float K_proj2 = dot(K_proj, K_proj);

                    // get polygon FT (may be accelerated in the future by converting to 2D)
                    float f2D_Im(0.0f);
                    float f2D_Re(0.0f);
                    // FT evaluated at K_proj==0 is the scattering volume (area)
                    if (K_proj2 == 0.0f)
                        {
                        f2D_Re = f2D_Im = m_params.area[facet_idx];
                        }
                    else
                        {
                        // f2D = -i/k^2 * \sum_0^{Nfacets - 1) \hat(z) \cdot (l_n \times k) \exp(-ik \cdot c_n) \sinc (k \cdot l/2)
                        // Noting that -i \exp(-i x) == \sin(x) - i \cos(x), we can get the real and imarginary parts as
                        // For each element in the sum,
                        // f_n = \hat(z) \cdot (l_n \times k) \sinc (k \cdot l/2) / k^2
                        // f_Re = \sin(k \cdot c_n) * f_n
                        // f_Im = - \cos(k \cdot c_n) * f_n
                        unsigned int N_vert = m_params.facet[facet_idx].size();
                        float f_n(0.0f);
                        float K2inv = 1.0f/K_proj2;
                        for(unsigned int edge_idx=0; edge_idx < N_vert; edge_idx++)
                            {
                            vec3<float> r0 = m_params.vert[m_params.facet[facet_idx][edge_idx]];
                            vec3<float> r1 = m_params.vert[m_params.facet[facet_idx][edge_idx+1]];
                            vec3<float> l_n = r1 - r0;
                            vec3<float> c_n = (r1 + r0)*0.5f;
                            float dotKc = dot(K_proj, c_n);
                            float dotKl = dot(K_proj, l_n);
                            vec3<float> crosslK = cross(l_n, K_proj);

                            float x = dotKl*0.5f; // argument to sinc function
                            f_n = dot(norm, crosslK) * (sinf(x)/x) * K2inv;

                            f2D_Re += sinf(dotKc) * f_n;
                            f2D_Im -= cosf(dotKc) * f_n;
                            } // end foreach edge
                        }

                    // accumulate
                    f_Re += f2D_Re;
                    f_Im += f2D_Im;
                    } // end foreach facet

                } // end K != 0

            // Get structure factor
            float CosKr, negSinKr; // real and (negative) imaginary components of exp(-i K r)
            float d; // dot product of K and r
            d = K.x * r.x + K.y * r.y + K.z * r.z;
            CosKr = cosf(d);
            negSinKr = sinf(d);

            // S += rho * f * exp(-i K r)
            S_Re += CosKr * f_Re + negSinKr * f_Im;
            S_Im += CosKr * f_Im - negSinKr * f_Re;
            } // end foreach particle

        m_S_Re[K_idx] = S_Re * rho_Re - S_Im * rho_Im;
        m_S_Im[K_idx] = S_Re * rho_Im + S_Im * rho_Re;
        } // end foreach K point
    }

void FTpolyhedron::set_params(const FTpolyhedron::param_type& params)
    {
    /*
    unsigned int N_vert = params.vert.size();
    m_params.vert.resize(N_vert);
    unsigned int N_facet = params.facet.size();
    m_params.facet.resize(N_facet);
    m_params.area.resize(N_facet);
    m_params.norm.resize(N_facet);
    m_params.d.resize(N_facet);
    */
    m_params = params;
    }

//! Helper function to build FTpolyhedron parameters
/*  \param vert list of (x,y,z) tuples
    \param facet list of lists of vertex indices
    \param norm list of (x,y,z) tuples of facet normal vectors
    \param d list of distances of facets from particle origin
    \param area list of facet areas
    \param volume particle volume
*/
poly3d_param_t make_poly3d(boost::python::list vert,
                           boost::python::list facet,
                           boost::python::list norm,
                           boost::python::list d,
                           boost::python::list area,
                           float volume)
    {
    poly3d_param_t result;
    result.volume = volume;
    unsigned int N_vert = len(vert);
    result.vert.resize(N_vert);
    for (unsigned int i=0; i < N_vert; i++)
        {
        boost::python::tuple v = extract<boost::python::tuple>(vert[i]);
        result.vert[i] = vec3<float>(extract<float>(v[0]), extract<float>(v[1]), extract<float>(v[2]));
        }

    unsigned int N_facet = len(facet);
    result.facet.resize(N_facet);
    result.norm.resize(N_facet);
    result.d.resize(N_facet);
    result.area.resize(N_facet);
    for (unsigned int i=0; i < N_facet; i++)
        {
        boost::python::list f = extract<boost::python::list>(facet[i]);
        result.facet[i].resize(len(f));
        for (unsigned int j=0 ; j < len(f); j++)
            {
            result.facet[i][j] = extract<unsigned int>(f[j]);
            }
        boost::python::tuple v = extract<boost::python::tuple>(norm[i]);
        result.norm[i] = vec3<float>(extract<float>(v[0]), extract<float>(v[1]), extract<float>(v[2]));
        result.d[i] = extract<float>(d[i]);
        result.area[i] = extract<float>(area[i]);
        }

    return result;
    }

void export_kspace()
    {
    class_<FTdelta>("FTdelta")
        .def("compute", &FTdelta::computePy)
        .def("getFT", &FTdelta::getFTPy)
        .def("set_K", &FTdelta::set_K_Py)
        .def("set_rq", &FTdelta::set_rq_Py)
        .def("set_density", &FTdelta::set_density)
        ;
    class_<FTsphere, bases<FTdelta> >("FTsphere")
        .def("set_radius", &FTsphere::set_radius)
        ;
    class_<FTpolyhedron, bases<FTdelta> >("FTpolyhedron")
        .def("set_params", &FTpolyhedron::set_params)
        ;
    class_<poly3d_param_t>("poly3d_param");
    def("make_poly3d", &make_poly3d);
    }

}; }; // end namespace freud::kspace
