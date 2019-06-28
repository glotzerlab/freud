// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include "PMFT.h"

using namespace std;
using namespace tbb;

/*! \internal
    \file PMFT.cc
    \brief Contains code for PMFT class
*/

namespace freud { namespace pmft {
/*! Initialize box
 */
PMFT::PMFT() : util::NdHistogram() {}

/*! All PMFT classes have the same deletion logic
 */

std::shared_ptr<float> PMFT::precomputeAxisBinCenter(unsigned int size, float d, float max)
{
    return precomputeArrayGeneral(size, d, [=](float T, float nextT) { return -max + ((T + nextT) / 2.0); });
}

}; }; // end namespace freud::pmft
