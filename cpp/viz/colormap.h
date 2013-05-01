#include <boost/python.hpp>

#include "HOOMDMath.h"
#include "Index1D.h"
#include "num_util.h"

#ifndef _COLORMAP_H__
#define _COLORMAP_H__

namespace freud { namespace colormap {

/*! \internal
    \brief Python wrapper for hue2RGBA
*/
void hue2RGBAPy(boost::python::numeric::array cmap, boost::python::numeric::array u, float a);

/*! \internal
    \brief Helper function for HSV to RGB conversion
*/
void hue2RGBA(float4 *cmap, const float *u, float a, unsigned int N);


/*! \internal
    \brief Exports all classes in this file to python 
*/
void export_colormap();
    
} } // end namespace freud::colormap

#endif
