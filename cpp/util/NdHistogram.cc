#include "NdHistogram.h"

namespace freud { namespace util {

NdHistogram::NdHistogram() : m_box(box::Box()), m_frame_counter(0), m_n_ref(0), m_n_p(0), m_reduce(true) {}

void NdHistogram::resetGeneral(unsigned int bin_size)
{
    for (tbb::enumerable_thread_specific<unsigned int*>::iterator i = m_local_bin_counts.begin();
         i != m_local_bin_counts.end(); ++i)
    {
        memset((void*) (*i), 0, sizeof(unsigned int) * bin_size);
    }
    this->m_frame_counter = 0;
    this->m_reduce = true;
}

}; };
