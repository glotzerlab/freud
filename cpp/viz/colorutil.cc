#include <boost/python.hpp>
#include <stdexcept>

#include "num_util.h"
#include "colorutil.h"
#include "ScopedGILRelease.h"
#include "colorutil.ispc.h"

#include <iostream>
#include <tbb/tbb.h>

using namespace std;
using namespace boost::python;
using namespace tbb;

namespace freud { namespace viz {

/*! \internal
    \brief Python wrapper for linearToSRGBA
    
    \param cmap Input/Output colormap (Nx4 float32 array)
*/
void linearToSRGBAPy(boost::python::numeric::array cmap)
    {
    //validate input type and rank
    num_util::check_type(cmap, PyArray_FLOAT);
    num_util::check_rank(cmap, 2);
    
    // validate that the 2nd dimension is 4
    num_util::check_dim(cmap, 1, 4);
    unsigned int N = num_util::shape(cmap)[0];
    
    // get the raw data pointers and compute conversion
    float4* cmap_raw = (float4*) num_util::data(cmap);
    
        // compute the colormap with the GIL released
        {
        util::ScopedGILRelease gil;
        linearToSRGBA(cmap_raw, N);
        }
    }

//! \internal
/*! \brief Helper class for parallel computation in linearToSRGBA
*/
class ComputeLinearToSRGBA
    {
    private:
        float4 *m_cmap;
    public:
        ComputeLinearToSRGBA(float4 *cmap)
            : m_cmap(cmap)
            {
            }
        
        void operator()( const blocked_range<size_t>& r ) const
            {
            ispc::viz_linearToSRGBA((float*)m_cmap, r.begin(), r.end());
            }
    };
                

/*! \param cmap Input/Output colormap (Nx4 float32 array)
    \param N Number of entires in the map
*/
void linearToSRGBA(float4 *cmap,
                   unsigned int N)
    {
    static affinity_partitioner ap;
    parallel_for(blocked_range<size_t>(0,N,100), ComputeLinearToSRGBA(cmap), ap);
    }

void export_colorutil()
    {
    def("linearToSRGBA", &linearToSRGBAPy);
    }

}; }; // end namespace freud::viz
