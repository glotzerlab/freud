#include <tbb/tbb.h>
#include <ostream>

// work around nasty issue where python #defines isalpha, toupper, etc....
#undef __APPLE__
#include <Python.h>
#define __APPLE__

#include "HOOMDMath.h"
#include "Index1D.h"

#ifndef _COLORUTIL_H__
#define _COLORUTIL_H__

/*! \file colorutil.h
    \brief Misc color utility functions
*/

namespace freud { namespace viz {

/*! \internal
    \brief Helper function for linear to/from SRGBA conversion
*/
void linearToFromSRGBA(float4 *cmap, unsigned int N, float p);

} } // end namespace freud::viz

#endif
