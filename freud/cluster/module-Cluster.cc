// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/nb_defs.h>

namespace nb = nanobind;

namespace freud::cluster::detail {
// NOLINTBEGIN(misc-use-internal-linkage)
void export_Cluster(nb::module_& module);
void export_ClusterProperties(nb::module_& module);
// NOLINTEND(misc-use-internal-linkage)
} // namespace freud::cluster::detail

using namespace freud::cluster::detail;

NB_MODULE(_cluster,
          module) // NOLINT(misc-use-anonymous-namespace, modernize-avoid-c-arrays): caused by nanobind
{
    export_Cluster(module);
    export_ClusterProperties(module);
}
