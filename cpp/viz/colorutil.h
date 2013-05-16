#include <boost/python.hpp>

#include "HOOMDMath.h"
#include "Index1D.h"
#include "num_util.h"

#ifndef _COLORUTIL_H__
#define _COLORUTIL_H__

namespace freud { namespace colorutil {

/*! \internal
    \brief Helper function for linear to SRGBA conversion
*/
void linearToSRGBA(float4 *cmap, unsigned int N);

/*! \internal
    \brief Exports all classes in this file to python 
*/
void export_colorutil();
    
} } // end namespace freud::colorutil

#endif
