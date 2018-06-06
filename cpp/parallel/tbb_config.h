// Copyright (c) 2010-2018 The Regents of the University of Michigan
// This file is part of the freud project, released under the BSD 3-Clause License.

#include <ostream>
#include <tbb/tbb.h>

#ifndef _TBB_CONFIG_H__
#define _TBB_CONFIG_H__

/*! \file tbb_config.h
    \brief Helper functions to configure tbb
*/

namespace freud { namespace parallel {

//! Set the number of TBB threads
void setNumThreads(unsigned int N);

} } // end namespace freud::parallel

#endif
