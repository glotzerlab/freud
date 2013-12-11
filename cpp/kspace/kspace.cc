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
            float CosKr, SinKr; // real and (negative) imaginary components of exp(-i K r)
            CosKr = cos(d);
            SinKr = sin(d);
            // S += rho * exp(-i K r)
            m_S_Re[i] += CosKr * density_Re + SinKr * density_Im;
            m_S_Im[i] += CosKr * density_Im + SinKr * density_Re;
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
            // FT evaluated at K=0 is just the scattering volume
            // f(0) = volume
            // f(K) = (4.*pi*R) / K**2 * (sinc(K*R) - cos(K*R)))
            if (K[i].x == 0.0f && K[i].y == 0.0f && K[i].z == 0.0f)
                {
                f_Im *= m_volume;
                f_Re *= m_volume;
                }
            else
                {
                float K2 = K[i].x * K[i].x + K[i].y * K[i].y + K[i].z * K[i].z;
                float KR = sqrtf(K2) * radius;
                float f = 4.0f * M_PI * radius / K2 * (sinf(KR)/KR - cosf(KR));
                f_Im *= f;
                f_Re *= f;
                }

            // Get structure factor
            float CosKr, SinKr; // real and (negative) imaginary components of exp(-i K r)
            float d; // dot product of K and r
            d = K[i].x * r[j].x + K[i].y * r[j].y + K[i].z * r[j].z;
            CosKr = cos(d);
            SinKr = sin(d);

            // S += rho * f * exp(-i K r)
            m_S_Re[i] += CosKr * f_Re + SinKr * f_Im;
            m_S_Im[i] += CosKr * f_Im + SinKr * f_Re;
            }
        }
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
    }

}; }; // end namespace freud::kspace
