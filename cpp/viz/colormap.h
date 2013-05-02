#include <boost/python.hpp>

#include "HOOMDMath.h"
#include "Index1D.h"
#include "num_util.h"

#ifndef _COLORMAP_H__
#define _COLORMAP_H__

namespace freud { namespace colormap {

/*! \internal
    \brief Helper function for HSV to RGB conversion
*/
void hsv2RGBA(float4 *cmap,
              const float *theta_array,
              const float *s_array,
              const float *v_array,
              float a,
              unsigned int N);

/*! \internal
    \brief Exports all classes in this file to python 
*/
void export_colormap();
    
} } // end namespace freud::colormap

#endif
