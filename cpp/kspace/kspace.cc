#include "kspace.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#include <complex>

using namespace std;
using namespace boost::python;

namespace freud { namespace kspace {

FTdelta::FTdelta()
    : m_NK(0),
      m_Np(0),
      m_scale(1.0),
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
    float scale = m_scale;
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
            d *= scale;
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

void export_kspace()
    {
    class_<FTdelta>("FTdelta")
        .def("compute", &FTdelta::computePy)
        .def("getFT", &FTdelta::getFTPy)
        .def("set_K", &FTdelta::set_K_Py)
        .def("set_rq", &FTdelta::set_rq_Py)
        .def("set_scale", &FTdelta::set_scale)
        .def("set_density", &FTdelta::set_density)
        ;
    }

}; }; // end namespace freud::kspace
