// Copyright (c) 2010-2026 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/nb_defs.h>

namespace freud::util::detail {
void export_ManagedArray(nanobind::module_& module);
void export_Vector(nanobind::module_& module);
} // namespace freud::util::detail

using namespace freud::util::detail;

NB_MODULE(_util, module) // NOLINT: caused by nanobind
{
    export_ManagedArray(module);
    export_Vector(module);
}
