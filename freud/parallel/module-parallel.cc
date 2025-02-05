// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/nb_defs.h>

#include "tbb_config.h"

using namespace freud::parallel;

NB_MODULE(_parallel, module) // NOLINT(misc-use-anonymous-namespace): caused by nanobind
{
    module.def("setNumThreads", &setNumThreads);
}
