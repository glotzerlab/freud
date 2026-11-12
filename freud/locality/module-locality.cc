// Copyright (c) 2010-2026 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/nb_defs.h>

#include "NeighborQuery.h"

namespace freud::locality::detail {

void export_Voronoi(nanobind::module_& module);
void export_PeriodicBuffer(nanobind::module_& module);
void export_NeighborQuery(nanobind::module_& module);
void export_AABBQuery(nanobind::module_& module);
void export_LinkCell(nanobind::module_& module);
void export_RawPoints(nanobind::module_& module);
void export_CellQuery(nanobind::module_& module);
void export_QueryArgs(nanobind::module_& module);
void export_NeighborQueryIterator(nanobind::module_& module);
void export_NeighborList(nanobind::module_& module);
void export_NeighborBond(nanobind::module_& module);
void export_BondHistogramCompute(nanobind::module_& module);
void export_Filter(nanobind::module_& module);
void export_FilterRAD(nanobind::module_& module);
void export_FilterSANN(nanobind::module_& module);
} // namespace freud::locality::detail

using namespace freud::locality::detail;

NB_MODULE(_locality, module) // NOLINT: caused by nanobind
{
    // for using ITERATOR_TERMINATOR at the python level
    module.def("get_iterator_terminator", []() { return freud::locality::ITERATOR_TERMINATOR; });

    // Neighbor finding stuff
    export_NeighborQuery(module);
    export_AABBQuery(module);
    export_LinkCell(module);
    export_RawPoints(module);
    export_CellQuery(module);
    export_QueryArgs(module);
    export_NeighborQueryIterator(module);
    export_NeighborList(module);
    export_NeighborBond(module);
    export_Voronoi(module);

    // filters
    export_Filter(module);
    export_FilterRAD(module);
    export_FilterSANN(module);

    // others
    export_PeriodicBuffer(module);
    export_BondHistogramCompute(module);
}
