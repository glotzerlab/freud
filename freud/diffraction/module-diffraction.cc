// Copyright (c) 2010-2026 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/nb_defs.h>

namespace freud::diffraction::detail {

void export_StaticStructureFactor(nanobind::module_& m);
void export_StaticStructureFactorDebye(nanobind::module_& m);
void export_StaticStructureFactorDirect(nanobind::module_& m);
} // namespace freud::diffraction::detail

using namespace freud::diffraction::detail;

NB_MODULE(_diffraction, module) // NOLINT: caused by nanobind
{
    export_StaticStructureFactor(module);
    export_StaticStructureFactorDebye(module);
    export_StaticStructureFactorDirect(module);
}
