#include <boost/python.hpp>

#include "HOOMDMath.h"
#include "Index1D.h"
#include "num_util.h"

#ifndef _COLORUTIL_H__
#define _COLORUTIL_H__

namespace freud { namespace viz {

/*! \internal
    \brief Helper function for linear to/from SRGBA conversion
*/
void linearToFromSRGBA(float4 *cmap, unsigned int N, float p);

/*! \internal
    \brief Exports all classes in this file to python 
*/
void export_colorutil();
    
} } // end namespace freud::viz

#endif
