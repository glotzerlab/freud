// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/shared_ptr.h>  // NOLINT(misc-include-cleaner): used implicitly
#include <nanobind/nanobind.h>

#include "Box.h"
#include "NeighborBond.h"
#include "NeighborList.h"
#include "VectorMath.h"

namespace nb = nanobind;

namespace freud { namespace locality {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap {

void ConstructFromArrays(NeighborList* nlist, const nb_array<unsigned int, nb::ndim<1>>& query_point_indices,
                         unsigned int num_query_points, const nb_array<unsigned int, nb::ndim<1>>& point_indices,
                         unsigned int num_points, const nb_array<float, nb::shape<-1, 3>>& vectors,
                         const nb_array<float, nb::ndim<1>>& weights)
{
    const unsigned int num_bonds = query_point_indices.shape(0);
    const auto* query_point_indices_data = (const unsigned int*) query_point_indices.data();
    const auto* point_indices_data = (const unsigned int*) point_indices.data();
    const auto* vectors_data = (const vec3<float>*) vectors.data();
    const auto* weights_data = (const float*) weights.data();
    new (nlist) NeighborList(num_bonds, query_point_indices_data, num_query_points, point_indices_data,
                             num_points, vectors_data, weights_data);
}

void ConstructAllPairs(NeighborList* nlist, const nb_array<float, nb::shape<-1, 3>>& points,
                       const nb_array<float, nb::shape<-1, 3>>& query_points, const box::Box& box,
                       const bool exclude_ii)
{
    const unsigned int num_points = points.shape(0);
    const auto* points_data = (const vec3<float>*) points.data();
    const unsigned int num_query_points = query_points.shape(0);
    const auto* query_points_data = (const vec3<float>*) query_points.data();
    new (nlist) NeighborList(points_data, query_points_data, box, exclude_ii, num_points, num_query_points);
}

unsigned int filter(const std::shared_ptr<NeighborList>& nlist, const nb_array<bool, nb::ndim<1>>& filter)
{
    const bool* filter_data = (const bool*) filter.data();
    return nlist->filter(filter_data);
}

}; // end namespace wrap

namespace detail {

void export_NeighborList(nb::module_& module)
{
    nb::class_<NeighborList>(module, "NeighborList")
        // export constructors, wrap some as static factory methods
        .def(nb::init<>())
        .def(nb::init<unsigned int>())
        .def("__init__", &wrap::ConstructFromArrays)
        .def("__init__", &wrap::ConstructAllPairs)
        // getters and setters
        .def("getNeighbors", &NeighborList::getNeighbors)
        .def("getWeights", &NeighborList::getWeights)
        .def("getDistances", &NeighborList::getDistances)
        .def("getVectors", &NeighborList::getVectors)
        .def("getSegments", &NeighborList::getSegments)
        .def("getCounts", &NeighborList::getCounts)
        .def("getNumBonds", &NeighborList::getNumBonds)
        .def("getNumQueryPoints", &NeighborList::getNumQueryPoints)
        .def("getNumPoints", &NeighborList::getNumPoints)
        // other member functions
        .def("copy", &NeighborList::copy)
        .def("find_first_index", &NeighborList::find_first_index)
        .def("filter", &wrap::filter)
        .def("filter_r", &NeighborList::filter_r)
        .def("sort", &NeighborList::sort);
};

void export_NeighborBond(nb::module_& module)
{
    nb::class_<NeighborBond>(module, "NeighborBond")
        .def(nb::init<>())
        .def("getQueryPointIdx", &NeighborBond::getQueryPointIdx)
        .def("getPointIdx", &NeighborBond::getPointIdx)
        .def("getDistance", &NeighborBond::getDistance)
        .def(nb::self == nb::self)  // NOLINT(misc-redundant-expression): valid nanobind syntax
        .def(nb::self != nb::self); // NOLINT(misc-redundant-expression): valid nanobind syntax
    };

}; // namespace detail

}; }; // namespace freud::locality
