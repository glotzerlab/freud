// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/nb_defs.h>

namespace freud::diffraction::detail {

void export_StaticStructureFactor(nanobind::module_& m);
// void export_PMFTXY(nanobind::module_& m);
// void export_PMFTXYZ(nanobind::module_& m);
// void export_PMFTR12(nanobind::module_& m);
// void export_PMFTXYT(nanobind::module_& m);
} // namespace freud::diffraction::detail

using namespace freud::diffraction::detail;

NB_MODULE(_diffraction, module) // NOLINT(misc-use-anonymous-namespace): caused by nanobind
{
    export_StaticStructureFactor(module);
    // export_PMFTXY(module);
    // export_PMFTXYZ(module);
    // export_PMFTR12(module);
    // export_PMFTXYT(module);
}
