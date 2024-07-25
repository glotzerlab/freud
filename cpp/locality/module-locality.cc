// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>

#include "export_Filter.h"
#include "export_NeighborList.h"
#include "export_NeighborQuery.h"
#include "export_PeriodicBuffer.h"
#include "export_Voronoi.h"

using namespace freud::locality::detail;

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
}
