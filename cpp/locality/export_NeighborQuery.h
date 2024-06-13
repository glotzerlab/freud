#ifndef EXPORT_NEIGHBOR_QUERY_H
#define EXPORT_NEIGHBOR_QUERY_H

#include "NeighborQuery.h"

#include <nanobind/nanobind.h>
namespace nb = nanobind;

namespace freud { namespace locality {



namespace detail
{
void export_NeighborQuery(nb::module_& m)
{
    nb::class_<NeighborQuery>(m, "NeighborQuery")
        .def(nb::init<>())
        .def("query", &NeighborQuery::query)
        .def("getBox", &NeighborQuery::getBox)
        .def("getPoints", &NeighborQuery::getPoints);
}
};

}; };

#endif
