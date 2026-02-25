// Copyright (c) 2010-2026 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/nb_defs.h>

namespace freud::order::detail {
void export_Nematic(nanobind::module_& m);
void export_RotationalAutocorrelation(nanobind::module_& m);
void export_Steinhardt(nanobind::module_& m);
void export_SolidLiquid(nanobind::module_& m);
void export_ContinuousCoordination(nanobind::module_& m);
void export_Cubatic(nanobind::module_& m);
void export_HexaticTranslational(nanobind::module_& m);
} // namespace freud::order::detail

using namespace freud::order::detail;

NB_MODULE(_order, module) // NOLINT: caused by nanobind
{
    export_Nematic(module);
    export_RotationalAutocorrelation(module);
    export_Steinhardt(module);
    export_SolidLiquid(module);
    export_ContinuousCoordination(module);
    export_Cubatic(module);
    export_HexaticTranslational(module);
}
