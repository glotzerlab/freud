#ifndef EXPORT_NEIGHBOR_LIST_H
#define EXPORT_NEIGHBOR_LIST_H

#include "NeighborList.h"
#include "NeighborBond.h"

#include <nanobind/ndarray.h>
namespace nb = nanobind;


namespace freud { namespace locality {

template<typename T, typename shape = nb::shape<-1, 3>>
using nb_array = nb::ndarray<T, shape, nb::device::cpu, nb::c_contig>;

namespace wrap
{

std::shared_ptr<NeighborList> fromNumBonds(unsigned int num_bonds)
{
    return std::make_shared<NeighborList>(num_bonds);
}

std::shared_ptr<NeighborList> fromArrays(nb_array<unsigned int, nb::ndim<1>> query_point_indices,
        unsigned int num_query_points, nb_array<unsigned int, nb::ndim<1>> point_indices,
        unsigned int num_points, nb_array<float> vectors, nb_array<float, nb::ndim<1>> weights)
{
    const unsigned int num_bonds = query_point_indices.shape(0);
    const auto *query_point_indices_data = (const unsigned int *)query_point_indices.data();
    const auto *point_indices_data = (const unsigned int *)point_indices.data();
    const auto *vectors_data = (const vec3<float>*)vectors.data();
    const auto *weights_data = (const float*)weights.data();
    return std::make_shared<NeighborList>(num_bonds, query_point_indices_data,
            num_query_points, point_indices_data, num_points, vectors_data, weights_data);
}

std::shared_ptr<NeighborList> allPairs(nb_array<float> points, nb_array<float> query_points,
        const box::Box& box, const bool exclude_ii)
{
    const unsigned int num_points = points.shape(0);
    const auto *points_data = (const vec3<float> *)points.data();
    const unsigned int num_query_points = query_points.shape(0);
    const auto *query_points_data = (const vec3<float> *)query_points.data();
    return std::make_shared<NeighborList>(points_data, query_points_data, box,
            exclude_ii, num_points, num_query_points);
}

};  // end namespace wrap

namespace detail
{
void export_NeighborList(nb::module_& m)
{
    nb::class_<NeighborList>(m, "NeighborList")
        // export constructors, wrap some as static factory methods
        .def(nb::init<>())
        .def_static("fromNumBonds", &wrap::fromNumBonds)
        .def_static("fromArrays", &wrap::fromArrays)
        .def_static("allPairs", &wrap::allPairs);
};

void export_NeighborBond(nb::module_& m)
{
    nb::class_<NeighborBond>(m, "NeighborBond")
        .def(nb::init<>())
        .def("getQueryPointIdx", &NeighborBond::getQueryPointIdx)
        .def("getPointIdx", &NeighborBond::getPointIdx)
        .def("getDistance", &NeighborBond::getDistance);
};
};

}; };  // end namespace freud:locality

#endif
