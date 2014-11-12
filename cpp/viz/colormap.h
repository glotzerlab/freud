#include <tbb/tbb.h>
#include <ostream>

// work around nasty issue where python #defines isalpha, toupper, etc....
#undef __APPLE__
#include <Python.h>
#define __APPLE__

#include <boost/python.hpp>

#include "HOOMDMath.h"
#include "Index1D.h"
#include "num_util.h"

#ifndef _COLORMAP_H__
#define _COLORMAP_H__

/*! \file colormap.h
    \brief Colormap build routines
*/

namespace freud { namespace viz {

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
    \brief Helper function for jet colormap
*/
void jet(float4 *cmap,
         const float *u_array,
         float a,
         unsigned int N);

/*! \internal
    \brief Helper function for cubehelix colormap
*/
void cubehelix(float4 *cmap,
               const float *lambda_array,
               unsigned int N,
               float a,
               float s,
               float r,
               float h,
               float gamma,
               bool reverse);

/*! \internal
    \brief Exports all classes in this file to python
*/
void export_colormap();

} } // end namespace freud::viz

#endif
