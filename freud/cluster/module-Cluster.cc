// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/nb_defs.h>

#include "NeighborQuery.h"

namespace nb = nanobind;

namespace freud::Cluster::detail {
void export_Cluster(nb::module_& module);
void export_ClusterProperties(nb::module_& module);
} // freud::Cluster::detail

using namespace freud::Cluster::detail;

NB_MODULE(_cluster, module)
{
    export_Cluster(module);
    export_ClusterProperties(module);
}