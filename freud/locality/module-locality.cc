// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>

#include "NeighborQuery.h"

namespace freud::locality::detail {

void export_Voronoi(nanobind::module_& m);
void export_PeriodicBuffer(nanobind::module_& m);
void export_NeighborQuery(nanobind::module_& m);
void export_AABBQuery(nanobind::module_& m);
void export_LinkCell(nanobind::module_& m);
void export_RawPoints(nanobind::module_& m);
void export_QueryArgs(nanobind::module_& m);
void export_NeighborQueryIterator(nanobind::module_& m);
void export_NeighborList(nanobind::module_& m);
void export_NeighborBond(nanobind::module_& m);
void export_BondHistogramCompute(nanobind::module_& m);
void export_Filter(nanobind::module_& m);
void export_FilterRAD(nanobind::module_& m);
void export_FilterSANN(nanobind::module_& m);
} // namespace freud::locality::detail

using namespace freud::locality::detail;
namespace nb = nanobind;

NB_MODULE(_locality, m)
{
    // for using ITERATOR_TERMINATOR at the python level
    m.def("get_iterator_terminator", []() { return freud::locality::ITERATOR_TERMINATOR; });

    // Neighbor finding stuff
    export_NeighborQuery(m);
    export_AABBQuery(m);
    export_LinkCell(m);
    export_RawPoints(m);
    export_QueryArgs(m);
    export_NeighborQueryIterator(m);
    export_NeighborList(m);
    export_NeighborBond(m);
    export_Voronoi(m);

    // filters
    export_Filter(m);
    export_FilterRAD(m);
    export_FilterSANN(m);

    // others
    export_PeriodicBuffer(m);
    export_BondHistogramCompute(m);
}
