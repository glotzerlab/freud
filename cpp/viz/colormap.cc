#include <boost/python.hpp>
#include <stdexcept>

#include "num_util.h"
#include "colormap.h"
#include "ScopedGILRelease.h"

#include <iostream>

using namespace std;
using namespace boost::python;

namespace freud { namespace viz {

/*! \internal
    \brief Python wrapper for hue2RGBA
    
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
    for (unsigned int i = 0; i < N; i++)
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
        
        cmap[i].x = r;
        cmap[i].y = g;
        cmap[i].z = b;
        cmap[i].w = a;
        }
    }


void export_colormap()
    {
    def("hsv2RGBA", &hsv2RGBAPy);
    }

}; }; // end namespace freud::viz
