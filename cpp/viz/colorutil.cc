#include "colorutil.h"

#include <stdexcept>

#include "ScopedGILRelease.h"
// #include "colorutil.ispc.h"

#include <iostream>
#include <tbb/tbb.h>

using namespace std;
using namespace tbb;

/*! \file colorutil.cc
    \brief Misc color utility functions
*/

namespace freud { namespace viz {

// /*! \internal
//     \brief Python wrapper for linearToFromSRGBA

//     \param cmap Input/Output colormap (Nx4 float32 array)
//     \param p Power to raise components by
// */
// void linearToFromSRGBAPy(boost::python::numeric::array cmap, float p)
//     {
//     //validate input type and rank
//     num_util::check_type(cmap, NPY_FLOAT);
//     num_util::check_rank(cmap, 2);

//     // validate that the 2nd dimension is 4
//     num_util::check_dim(cmap, 1, 4);
//     unsigned int N = num_util::shape(cmap)[0];

//     // get the raw data pointers and compute conversion
//     float4* cmap_raw = (float4*) num_util::data(cmap);

//         // compute the colormap with the GIL released
//         {
//         util::ScopedGILRelease gil;
//         linearToFromSRGBA(cmap_raw, N, p);
//         }
//     }



/*! \param cmap Input/Output colormap (Nx4 float32 array)
    \param N Number of entires in the map
    \param p Power to raise components by
*/
void linearToFromSRGBA(float4 *cmap,
                       unsigned int N,
                       float p)
    {
    static affinity_partitioner ap;
    parallel_for(blocked_range<size_t>(0,N,100),
    [=] (const blocked_range<size_t>& r)
    {
    // ispc::viz_linearToSRGBA((float*)m_cmap, r.begin(), r.end(), m_p);
    for (unsigned int i = r.begin(); i < r.end(); i++)
        {
        cmap[i].x = powf(cmap[i].x, p);
        cmap[i].y = powf(cmap[i].y, p);
        cmap[i].z = powf(cmap[i].z, p);
        }
    });
    }

}; }; // end namespace freud::viz
