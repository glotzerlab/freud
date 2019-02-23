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
PMFT::PMFT()
    : m_box(box::Box()), m_frame_counter(0), m_n_ref(0), m_n_p(0), m_reduce(true)
    {
    }

/*! All PMFT classes have the same deletion logic
 */
PMFT::~PMFT()
    {
    for (tbb::enumerable_thread_specific<unsigned int *>::iterator i = m_local_bin_counts.begin(); i != m_local_bin_counts.end(); ++i)
        {
        delete[] (*i);
        }
    }

//! Get a reference to the PCF array
std::shared_ptr<unsigned int> PMFT::getBinCounts()
    {
    if (m_reduce == true)
        {
        reducePCF();
        }
    m_reduce = false;
    return m_bin_counts;
    }

//! Get a reference to the PCF array
std::shared_ptr<float> PMFT::getPCF()
    {
    if (m_reduce == true)
        {
        reducePCF();
        }
    m_reduce = false;
    return m_pcf_array;
    }

}; }; // end namespace freud::pmft
