// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/nb_defs.h>

namespace freud::pmft::detail {

void export_PMFT(nanobind::module_& m);
void export_PMFTXY(nanobind::module_& m);
void export_PMFTXYZ(nanobind::module_& m);
void export_PMFTR12(nanobind::module_& m);
void export_PMFTXYT(nanobind::module_& m);
} // namespace freud::pmft::detail

using namespace freud::pmft::detail;

NB_MODULE(_pmft, module) // NOLINT: caused by nanobind
{
    export_PMFT(module);
    export_PMFTXY(module);
    export_PMFTXYZ(module);
    export_PMFTR12(module);
    export_PMFTXYT(module);
}
