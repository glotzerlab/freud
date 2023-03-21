// Copyright (c) 2010-2023 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "tbb_config.h"

#include <thread>

/*! \file tbb_config.cc
    \brief Helper functions to configure tbb
*/

namespace freud { namespace parallel {

std::unique_ptr<tbb::global_control> tbb_thread_control;

/*! \param N Number of threads to use for TBB computations

    You do not need to call setTBBNumThreads. The default is to use the number of threads in the system. Use
   \a N=0 to set back to the default.

    \note setTBBNumThreads should only be called from the main thread.
*/
void setNumThreads(unsigned int N)
{
    if (N == 0)
    {
        N = std::thread::hardware_concurrency();
    }

    // then recreate it
    tbb_thread_control
        = std::make_unique<tbb::global_control>(tbb::global_control::parameter::max_allowed_parallelism, N);
}

}; }; // end namespace freud::parallel
