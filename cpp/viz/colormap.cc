#include <boost/python.hpp>
#include <stdexcept>

#include "num_util.h"
#include "colormap.h"
#include "ScopedGILRelease.h"

#include <iostream>
#include <tbb/tbb.h>

using namespace std;
using namespace boost::python;
using namespace tbb;

/*! \file colormap.cc
    \brief Colormap build routines
*/

namespace freud { namespace viz {

/*! \internal
    \brief Python wrapper for hsv2RGBA

    \param cmap Output colormap (Nx4 float32 array)
    \param theta Input values: hue angle (N element float32 array)
    \param s Input values: saturation (N element float32 array)
    \param v Input values: intensity (N element float32 array)
    \param a Alpha value
*/
void hsv2RGBAPy(boost::python::numeric::array cmap,
                boost::python::numeric::array theta,
                boost::python::numeric::array s,
                boost::python::numeric::array v,
                float a)
    {
    //validate input type and rank
    num_util::check_type(cmap, PyArray_FLOAT);
    num_util::check_rank(cmap, 2);

    // validate that the 2nd dimension is 4
    num_util::check_dim(cmap, 1, 4);
    unsigned int N = num_util::shape(cmap)[0];

    // check that u is consistent
    num_util::check_type(theta, PyArray_FLOAT);
    num_util::check_rank(theta, 1);
    if (num_util::shape(theta)[0] != N)
        throw std::invalid_argument("Input lengths for cmap and theta must match");

    // check that s is consistent
    num_util::check_type(s, PyArray_FLOAT);
    num_util::check_rank(s, 1);
    if (num_util::shape(s)[0] != N)
        throw std::invalid_argument("Input lengths for cmap and s must match");

    // check that v is consistent
    num_util::check_type(v, PyArray_FLOAT);
    num_util::check_rank(v, 1);
    if (num_util::shape(v)[0] != N)
        throw std::invalid_argument("Input lengths for cmap and v must match");

    // get the raw data pointers and compute conversion
    float4* cmap_raw = (float4*) num_util::data(cmap);
    float* theta_raw = (float*)num_util::data(theta);
    float* s_raw = (float*)num_util::data(s);
    float* v_raw = (float*)num_util::data(v);

        // compute the colormap with the GIL released
        {
        util::ScopedGILRelease gil;
        hsv2RGBA(cmap_raw, theta_raw, s_raw, v_raw, a, N);
        }
    }

//! \internal
/*! \brief Helper class for parallel computation in linearToSRGBA
*/
class ComputeHSV2RGBA
    {
    private:
        float4 *m_cmap;
        const float *m_theta_array;
        const float *m_s_array;
        const float *m_v_array;
        const float m_a;
    public:
        ComputeHSV2RGBA(float4 *cmap,
                        const float *theta_array,
                        const float *s_array,
                        const float *v_array,
                        const float a)
            : m_cmap(cmap), m_theta_array(theta_array), m_s_array(s_array), m_v_array(v_array), m_a(a)
            {
            }

        void operator()( const blocked_range<size_t>& r ) const
            {
            float4 *cmap = m_cmap;
            const float *theta_array = m_theta_array;
            const float *s_array = m_s_array;
            const float *v_array = m_v_array;
            const float a = m_a;

            for (size_t i = r.begin(); i < r.end(); ++i)
                {
                // algorithm from http://en.wikipedia.org/wiki/HSL_and_HSV
                float min;
                float chroma;
                float Hprime;
                float X;

                // fix s and V to 1.0 for now.
                float v = v_array[i];
                float s = s_array[i];

                // temporary holders for r,g,b
                float r=0.0f, g=0.0f, b=0.0f;

                // map angle to 0-2pi range
                float theta = fmodf(theta_array[i],M_PI*2.0f);
                if (theta < 0.0f)
                    theta += M_PI*2.0f;

                // compute rgb from hue angle
                chroma = s*v;
                Hprime = theta / (M_PI / 3.0f);
                X = chroma * (1.0f - fabsf(fmodf(Hprime, 2.0f) - 1.0f));

                if(Hprime < 1.0f)
                    {
                    r = chroma;
                    g = X;
                    }
                else if(Hprime < 2.0f)
                    {
                    r = X;
                    g = chroma;
                    }
                else if(Hprime < 3.0f)
                    {
                    g = chroma;
                    b = X;
                    }
                else if(Hprime < 4.0f)
                    {
                    g= X;
                    b = chroma;
                    }
                else if(Hprime < 5.0f)
                    {
                    r = X;
                    b = chroma;
                    }
                else if(Hprime <= 6.0f)
                    {
                    r = chroma;
                    b = X;
                    }

                min = v - chroma;

                r += min;
                g += min;
                b += min;

                cmap[i].x = powf(r, 1.0f/2.2f);
                cmap[i].y = powf(g, 1.0f/2.2f);
                cmap[i].z = powf(b, 1.0f/2.2f);
                cmap[i].w = a;
                }
            }
    };


/*! \param cmap Output colormap (Nx4 float32 array)
    \param theta_array Input values: hue angle (N element float32 array)
    \param s_array Input values: saturation (N element float32 array)
    \param v_array Input values: intensity (N element float32 array)
    \param a Alpha value
*/
void hsv2RGBA(float4 *cmap,
              const float *theta_array,
              const float *s_array,
              const float *v_array,
              float a,
              unsigned int N)
    {
    parallel_for(blocked_range<size_t>(0,N,100), ComputeHSV2RGBA(cmap, theta_array, s_array, v_array, a));
    }


/*! \internal
    \brief Python wrapper for jet

    \param cmap Output colormap (Nx4 float32 array)
    \param u Input values: linear in range 0-1 (N element float32 array)
    \param a Alpha value
*/
void jetPy(boost::python::numeric::array cmap,
           boost::python::numeric::array u,
           float a)
    {
    //validate input type and rank
    num_util::check_type(cmap, PyArray_FLOAT);
    num_util::check_rank(cmap, 2);

    // validate that the 2nd dimension is 4
    num_util::check_dim(cmap, 1, 4);
    unsigned int N = num_util::shape(cmap)[0];

    // check that u is consistent
    num_util::check_type(u, PyArray_FLOAT);
    num_util::check_rank(u, 1);
    if (num_util::shape(u)[0] != N)
        throw std::invalid_argument("Input lengths for cmap and u must match");

    // get the raw data pointers and compute conversion
    float4* cmap_raw = (float4*) num_util::data(cmap);
    float* u_raw = (float*)num_util::data(u);

        // compute the colormap with the GIL released
        {
        util::ScopedGILRelease gil;
        jet(cmap_raw, u_raw, a, N);
        }
    }

//! \internal
/*! \brief Helper class for parallel computation in jet()
*/
class ComputeJet
    {
    private:
        float4 *m_cmap;
        const float *m_u_array;
        const float m_a;
    public:
        ComputeJet(float4 *cmap,
                        const float *u_array,
                        const float a)
            : m_cmap(cmap), m_u_array(u_array), m_a(a)
            {
            }

        void operator()( const blocked_range<size_t>& r ) const
            {
            float4 *cmap = m_cmap;
            const float *u_array = m_u_array;

            for (size_t i = r.begin(); i < r.end(); ++i)
                {
                // clamp the input
                float u = u_array[i];
                u = max(0.0f, u);
                u = min(1.0f, u);

                // compute jet map
                float v = 4.0f * u_array[i];
                float r = min(v - 1.5f, -v + 4.5f);
                float g = min(v - 0.5f, -v + 3.5f);
                float b = min(v + 0.5f, -v + 2.5f);

                // clamp to range 0,1
                r = max(0.0f, r);
                r = min(1.0f, r);

                g = max(0.0f, g);
                g = min(1.0f, g);

                b = max(0.0f, b);
                b = min(1.0f, b);

                /*cmap[i].x = powf(r, 1.0f/2.2f);
                cmap[i].y = powf(g, 1.0f/2.2f);
                cmap[i].z = powf(b, 1.0f/2.2f);*/
                cmap[i].x = r;
                cmap[i].y = g;
                cmap[i].z = b;
                cmap[i].w = m_a;
                }
            }
    };


/*! \param cmap Output colormap (Nx4 float32 array)
    \param u_array Input values: linear values (N element float32 array)
    \param a Alpha value
*/
void jet(float4 *cmap,
         const float *u_array,
         float a,
         unsigned int N)
    {
    parallel_for(blocked_range<size_t>(0,N,100), ComputeJet(cmap, u_array, a));
    }

/*! \internal
    \brief Python wrapper for cubehelix

    \param cmap Output colormap (Nx4 float32 array)
    \param lambda Input values: linear in range 0-1 (N element float32 array)
    \param a Alpha value
    \param s Hue of the starting color
    \param r Number of rotations through R->G->B to make
    \param h Hue parameter controlling saturation
    \param gamma Reweighting power to emphasize low intensity values or high intensity values
    \param reverse Reverse the colormap (lambda -> 1 - lambda)
*/
void cubehelixPy(boost::python::numeric::array cmap,
           boost::python::numeric::array lambda,
           float a,
           float s,
           float r,
           float h,
           float gamma,
           bool reverse)
    {
    //validate input type and rank
    num_util::check_type(cmap, PyArray_FLOAT);
    num_util::check_rank(cmap, 2);

    // validate that the 2nd dimension is 4
    num_util::check_dim(cmap, 1, 4);
    unsigned int N = num_util::shape(cmap)[0];

    // check that lambda is consistent
    num_util::check_type(lambda, PyArray_FLOAT);
    num_util::check_rank(lambda, 1);
    if (num_util::shape(lambda)[0] != N)
        throw std::invalid_argument("Input lengths for cmap and lambda must match");

    // get the raw data pointers and compute conversion
    float4* cmap_raw = (float4*) num_util::data(cmap);
    float* lambda_raw = (float*)num_util::data(lambda);

        // compute the colormap with the GIL released
        {
        util::ScopedGILRelease gil;
        cubehelix(cmap_raw, lambda_raw, N, a, s, r, h, gamma, reverse);
        }
    }

//! \internal
/*! \brief Helper class for parallel computation in cubehelix()
*/
class ComputeCubehelix
    {
    private:
        float4 *m_cmap;
        const float *m_lambda_array;
        const float m_a;
        const float m_s;
        const float m_r;
        const float m_h;
        const float m_gamma;
        const bool m_reverse;
    public:
        ComputeCubehelix(float4 *cmap,
                        const float *lambda_array,
                        const float a,
                        const float s,
                        const float r,
                        const float h,
                        const float gamma,
                        const bool reverse)
            : m_cmap(cmap), m_lambda_array(lambda_array), m_a(a), m_s(s),
              m_r(r), m_h(h), m_gamma(gamma), m_reverse(reverse)
            {
            }

        void operator()( const blocked_range<size_t>& r ) const
            {
            float4 *cmap = m_cmap;
            const float *lambda_array = m_lambda_array;

            for (size_t i = r.begin(); i < r.end(); ++i)
                {
                // clamp the input
                float lambda = lambda_array[i];
                lambda = max(0.0f, lambda);
                lambda = min(1.0f, lambda);
                if (m_reverse)
                    lambda = 1.0f - lambda;
                lambda = powf(lambda, m_gamma);

                const float phi = 2*M_PI*(m_s*(1.0f/3.0f) + m_r*lambda);
                // Note that this is the "a" parameter from the paper
                // and has nothing to do with m_a (the alpha value of
                // the color to return)
                const float a = m_h*lambda*(1.0f - lambda)*0.5f;

                // sin and cosine of phi
                const float sphi = sinf(phi);
                const float cphi = cosf(phi);

                float r = lambda + a*(-0.14861f*cphi + 1.78277f*sphi);
                float g = lambda + a*(-0.29227f*cphi - 0.90649f*sphi);
                float b = lambda + a*( 1.97294f*cphi);

                // clamp to range 0,1
                r = max(0.0f, r);
                r = min(1.0f, r);

                g = max(0.0f, g);
                g = min(1.0f, g);

                b = max(0.0f, b);
                b = min(1.0f, b);

                cmap[i].x = r;
                cmap[i].y = g;
                cmap[i].z = b;
                cmap[i].w = m_a;
                }
            }
    };


/*! \param cmap Output colormap (Nx4 float32 array)
    \param lambda_array Input values: linear values (N element float32 array)
    \param a Alpha value
    \param s Hue of the starting color
    \param r Number of rotations through R->G->B to make
    \param h Hue parameter controlling saturation
    \param gamma Reweighting power to emphasize low intensity values or high intensity values
    \param reverse Reverse the colormap (lambda -> 1 - lambda)
*/
void cubehelix(float4 *cmap,
         const float *lambda_array,
               unsigned int N,
               float a,
               float s,
               float r,
               float h,
               float gamma,
               bool reverse)
    {
    parallel_for(blocked_range<size_t>(0,N,100), ComputeCubehelix(cmap, lambda_array, a, s, r, h, gamma, reverse));
    }


void export_colormap()
    {
    def("hsv2RGBA", &hsv2RGBAPy);
    def("jet", &jetPy);
    def("cubehelix", &cubehelixPy);
    }

}; }; // end namespace freud::viz
