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

//! \internal
/*! \brief Helper class for parallel computation in linearToSRGBA
*/
class ComputeLinearToFromSRGBA
    {
    private:
        float4 *m_cmap;
        float m_p;
    public:
        ComputeLinearToFromSRGBA(float4 *cmap, float p)
            : m_cmap(cmap), m_p(p)
            {
            }

        void operator()( const blocked_range<size_t>& r ) const
            {
            // ispc::viz_linearToSRGBA((float*)m_cmap, r.begin(), r.end(), m_p);
            for (unsigned int i = r.begin(); i < r.end(); i++)
                {
                m_cmap[i].x = powf(m_cmap[i].x, m_p);
                m_cmap[i].y = powf(m_cmap[i].y, m_p);
                m_cmap[i].z = powf(m_cmap[i].z, m_p);
                }
            }
    };


/*! \param cmap Input/Output colormap (Nx4 float32 array)
    \param N Number of entires in the map
    \param p Power to raise components by
*/
void linearToFromSRGBA(float4 *cmap,
                       unsigned int N,
                       float p)
    {
    static affinity_partitioner ap;
    parallel_for(blocked_range<size_t>(0,N,100), ComputeLinearToFromSRGBA(cmap, p), ap);
    }

}; }; // end namespace freud::viz
