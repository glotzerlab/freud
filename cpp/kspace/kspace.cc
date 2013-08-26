#include "kspace.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#include <complex>

using namespace std;
using namespace boost::python;

namespace freud { namespace kspace {

FTdelta::FTdelta()
    {
    }

FTdelta::~FTdelta()
    {
    // S_Re and S_Im are boost::shared_array which need to be passed to Python and which should clean up after themselves
    }

void FTdelta::compute(const float3 *K,
                 const unsigned int NK,
                 const float3 *r,
                 const float4 *q,
                 const unsigned int Np,
                 const float scale,
                 const float density_Re,
                 const float density_Im
                 )
    {
    /* S += e**(-i * dot(K, r))
       -> S_Re += cos(dot(K, r))
       -> S_Im += - sin(dot(K, r))
    */
    m_NK = NK;
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

void FTdelta::computePy(boost::python::numeric::array K,
                        boost::python::numeric::array r,
                        boost::python::numeric::array q,
                        const float scale,
                        const std::complex<float> density
                        )
    {
    // validate input type and rank
    num_util::check_type(K, PyArray_FLOAT);
    num_util::check_rank(K, 2);
    num_util::check_type(r, PyArray_FLOAT);
    num_util::check_rank(r, 2);
    num_util::check_type(q, PyArray_FLOAT);
    num_util::check_rank(q, 2);

    // validate width of the 2nd dimension
    num_util::check_dim(K, 1, 3);
    unsigned int NK = num_util::shape(K)[0];

    num_util::check_dim(r, 1, 3);
    unsigned int Np = num_util::shape(r)[0];

    num_util::check_dim(q, 1, 4);
    num_util::check_dim(q, 0, Np);

    // get the raw data pointers
    float3* K_raw = (float3*) num_util::data(K);
    float3* r_raw = (float3*) num_util::data(r);
    float4* q_raw = (float4*) num_util::data(q);

        // compute with the GIL released
        {
        util::ScopedGILRelease gil;
        compute(K_raw, NK, r_raw, q_raw, Np, scale, density.real(), density.imag());
        }
    }

void export_kspace()
    {
    class_<FTdelta>("FTdelta")
        .def("compute", &FTdelta::computePy)
        .def("getFT", &FTdelta::getFTPy)
        ;
    }

}; }; // end namespace freud::kspace
