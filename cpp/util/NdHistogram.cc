#include "NdHistogram.h"

namespace freud { namespace util {

NdHistogram::NdHistogram() : m_box(box::Box()), m_frame_counter(0), m_n_ref(0), m_n_p(0), m_reduce(true) {}

void NdHistogram::resetGeneral(unsigned int bin_size)
{
    m_local_bin_counts.reset();
    this->m_frame_counter = 0;
    this->m_reduce = true;
}

}; }; // namespace freud::util
