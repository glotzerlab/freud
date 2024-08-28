// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>

namespace freud::pmft::detail {
    
void export_PMFT(nanobind::module_& m);
void export_PMFTXY(nanobind::module_& m);
void export_PMFTXYZ(nanobind::module_& m);
void export_PMFTR12(nanobind::module_& m);
void export_PMFTXYT(nanobind::module_& m);
}

using namespace freud::pmft::detail;

NB_MODULE(_pmft, m)
{
    export_PMFT(m);
    export_PMFTXY(m);
    export_PMFTXYZ(m);
    export_PMFTR12(m);
    export_PMFTXYT(m);
}
