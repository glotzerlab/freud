#include "NdHistogram.h"

namespace freud { namespace util {

void NdHistogram::reset() 
{
    // zero the bin counts for totaling
    for (tbb::enumerable_thread_specific<unsigned int *>::iterator i = m_local_bin_counts.begin(); i != m_local_bin_counts.end(); ++i)
        {
        memset((void*)(*i), 0, sizeof(unsigned int)*m_nbins);
        }
    for (typename tbb::enumerable_thread_specific<T *>::iterator i = m_local_rdf_array.begin(); i != m_local_rdf_array.end(); ++i)
        {
        memset((void*)(*i), 0, sizeof(T)*m_nbins);
        }
    // reset the frame counter
    m_frame_counter = 0;
    m_reduce = true;
}

template <typename Func>
void accumulateGeneral(box::Box& box, 
                       const freud::locality::NeighborList *nlist,
                       const vec3<float> *ref_points,
                       unsigned int n_ref,
                       const vec3<float> *points,
                       unsigned int Np, Func fn);

}; };
