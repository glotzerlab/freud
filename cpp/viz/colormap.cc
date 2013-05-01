#include <boost/python.hpp>
#include <stdexcept>

#include "num_util.h"
#include "colormap.h"
#include "ScopedGILRelease.h"

#include <iostream>

using namespace std;
using namespace boost::python;

namespace freud { namespace colormap {

/*! \param cmap Output colormap (Nx4 float32 array)
    \param u Input values (N element float32 array)
    \param a Alpha value
*/
void hue2RGBAPy(boost::python::numeric::array cmap, boost::python::numeric::array u, float a)
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
    hue2RGBA(cmap_raw, u_raw, a, N);
    }

/*! \param cmap Output colormap (Nx4 float32 array)
    \param u Input values (N element float32 array)
    \param a Alpha value
*/
void hue2RGBA(float4 *cmap, const float *u, float a, unsigned int N)
    {
    util::ScopedGILRelease();
    for (unsigned int i = 0; i < N; i++)
        {
        // algorithm from http://en.wikipedia.org/wiki/HSL_and_HSV
        float min;
        float chroma;
        float Hprime;
        float X;
        
        // fix s and V to 1.0 for now.
        float v = 1.0f;
        float s = 1.0f;
        
        // temporary holders for r,g,b
        float r=0.0f, g=0.0f, b=0.0f;
             
        chroma = s*v;
        Hprime = u[i] / (M_PI / 3.0f);
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
    def("hue2RGBA", &hue2RGBAPy);
    }

}; }; // end namespace freud::colormap
