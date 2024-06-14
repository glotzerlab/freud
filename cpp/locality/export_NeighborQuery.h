#ifndef EXPORT_NEIGHBOR_QUERY_H
#define EXPORT_NEIGHBOR_QUERY_H

#include "NeighborQuery.h"
#include "AABBQuery.h"
#include "LinkCell.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
namespace nb = nanobind;

namespace freud { namespace locality {

template<typename T, typename shape = nanobind::shape<-1, 3>>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap
{

std::shared_ptr<NeighborQueryIterator> query(std::shared_ptr<NeighborQuery> nq,
        nb_array<float> query_points, const QueryArgs& qargs)
{
    unsigned int n_query_points = query_points.shape(0);
    const vec3<float>* query_points_data = (vec3<float>*) query_points.data();
    return nq->query(query_points_data, n_query_points, qargs);
}

void AABBQueryConstructor(AABBQuery* nq, const box::Box& box, nb_array<float> points)
{
    unsigned int n_points = points.shape(0);
    vec3<float>* points_data = (vec3<float>*)points.data();
    new (nq) AABBQuery(box, points_data, n_points);
}

void LinkCellConstructor(LinkCell* nq, const box::Box& box, nb_array<float> points, float cell_width)
{
    unsigned int n_points = points.shape(0);
    vec3<float>* points_data = (vec3<float>*)points.data();
    new (nq) LinkCell(box, points_data, n_points, cell_width);
}

void RawPointsConstructor(LinkCell* nq, const box::Box& box, nb_array<float> points)
{
    unsigned int n_points = points.shape(0);
    vec3<float>* points_data = (vec3<float>*)points.data();
    new (nq) RawPoints(box, points_data, n_points);
}

};

namespace detail
{
void export_NeighborQuery(nb::module_& m)
{
    nb::class_<NeighborQuery>(m, "NeighborQuery")
        .def("query", &wrap::query)
        .def("getBox", &NeighborQuery::getBox)
}

void export_AABBQuery(nb::module_& m)
{
    nb::class_<AABBQuery, NeighborQuery>(m, "AABBQuery")
        .def("__init__", &wrap::AABBQueryConstructor);
}

void export_LinkCell(nb::module_& m)
{
    nb::class_<LinkCell, NeighborQuery>(m, "LinkCell")
        .def("__init__", &wrap::LinkCellConstructor)
        .def("GetCellWidth", &LinkCell::getCellWidth);
}

void export_RawPoints(nb::module_& m)
{
    nb::class_<RawPoints, NeighborQuery>(m, "LinkCell")
        .def("__init__", &wrap::RawPointsConstructor);
}
};

}; };

#endif
