// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/nb_defs.h>

namespace freud::environment::detail {

void export_AngularSeparationNeighbor(nanobind::module_& m);
void export_AngularSeparationGlobal(nanobind::module_& m);
void export_LocalBondProjection(nanobind::module_& m);
}; // namespace freud::environment::detail
using namespace freud::environment::detail;

NB_MODULE(_environment, module) // NOLINT(misc-use-anonymous-namespace): caused by nanobind
{
    export_AngularSeparationNeighbor(module);
    export_AngularSeparationGlobal(module);
    export_LocalBondProjection(module);
}
