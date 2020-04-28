// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef TBB_CONFIG_H
#define TBB_CONFIG_H

#include <tbb/task_scheduler_init.h>

/*! \file tbb_config.h
    \brief Helper functions to configure tbb
*/

namespace freud { namespace parallel {

//! Set the number of TBB threads
void setNumThreads(unsigned int N);

}; }; // end namespace freud::parallel

#endif // TBB_CONFIG_H
