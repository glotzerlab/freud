// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly

#include "LocalDensity.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "VectorMath.h"

namespace freud { namespace density {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap {

void computeLocalDensity(const std::shared_ptr<LocalDensity>& self,
                         const std::shared_ptr<locality::NeighborQuery>& points,
                         nb_array<float, nanobind::shape<-1, 3>>& query_points,
                         const unsigned int num_query_points, std::shared_ptr<locality::NeighborList>& nlist,
                         const locality::QueryArgs& qargs)
{
    auto* query_points_data = reinterpret_cast<vec3<float>*>(query_points.data());
    self->compute(points, query_points_data, num_query_points, nlist, qargs);
}

}; // namespace wrap

namespace detail {

void export_LocalDensity(nanobind::module_& m)
{
    nanobind::class_<LocalDensity>(m, "LocalDensity")
        .def(nanobind::init<float, float>())
        .def("compute", &wrap::computeLocalDensity, nanobind::arg("points"),
             nanobind::arg("query_points").none(), nanobind::arg("num_query_points").none(),
             nanobind::arg("nlist").none(), nanobind::arg("qargs").none())
        .def("getRMax", &LocalDensity::getRMax)
        .def("getDiameter", &LocalDensity::getDiameter)
        .def_prop_ro("box", &LocalDensity::getBox)
        .def_prop_ro("density", &LocalDensity::getDensity)
        .def_prop_ro("num_neighbors", &LocalDensity::getNumNeighbors);
}

} // namespace detail

}; }; // namespace freud::density
