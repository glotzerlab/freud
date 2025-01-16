// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/nb_defs.h>

namespace freud::density::detail {

// void export_CorrelationFunction(nanobind::module_& m);
void export_GaussianDensity(nanobind::module_& m);
void export_RDF(nanobind::module_& m);
void export_LocalDensity(nanobind::module_& m);
void export_SphereVoxelization(nanobind::module_& m);
} // namespace freud::density::detail

using namespace freud::density::detail;

NB_MODULE(_density, module) // NOLINT(misc-use-anonymous-namespace): caused by nanobind
{
    export_RDF(module);
    export_GaussianDensity(module);
    // export_CorrelationFunction(module);
    export_LocalDensity(module);
    export_SphereVoxelization(module);
}
