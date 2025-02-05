// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly
#include <nanobind/stl/vector.h>     // NOLINT(misc-include-cleaner): used implicitly

#include "AngularSeparation.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "VectorMath.h"

namespace nb = nanobind;

namespace freud { namespace environment {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap {

void compute(std::shared_ptr<AngularSeparationNeighbor>& angular_separation,
             const std::shared_ptr<locality::NeighborQuery>& nq,
             nb_array<float, nanobind::shape<-1, 4>>& orientations,
             nb_array<float, nanobind::shape<-1, 3>>& query_points,
             nb_array<float, nanobind::shape<-1, 4>>& query_orientations,
             nb_array<float, nanobind::shape<-1, 4>>& equiv_orientations,
             const std::shared_ptr<locality::NeighborList>& nlist, const locality::QueryArgs& qargs)
{
    auto* orientations_data = reinterpret_cast<quat<float>*>(orientations.data());
    auto* query_points_data = reinterpret_cast<vec3<float>*>(query_points.data());
    auto* query_orientations_data = reinterpret_cast<quat<float>*>(query_orientations.data());
    auto* equiv_orientations_data = reinterpret_cast<quat<float>*>(equiv_orientations.data());
    const unsigned int n_equiv_orientations = equiv_orientations.shape(0);
    const unsigned int n_query_points = query_orientations.shape(0);
    angular_separation->compute(nq, orientations_data, query_points_data, query_orientations_data,
                                n_query_points, equiv_orientations_data, n_equiv_orientations, nlist, qargs);
}
}; // namespace wrap

namespace detail {

void export_AngularSeparationNeighbor(nb::module_& module)
{
    nb::class_<AngularSeparationNeighbor>(module, "AngularSeparationNeighbor")
        .def(nb::init<>())
        .def("getNList", &AngularSeparationNeighbor::getNList)
        .def("getAngles", &AngularSeparationNeighbor::getAngles)
        .def("compute", &wrap::compute, nb::arg("nq"), nb::arg("orientations"), nb::arg("query_points"),
             nb::arg("query_orientations"), nb::arg("equiv_orientations"), nb::arg("nlist").none(),
             nb::arg("qargs"));
}

}; // namespace detail

}; }; // namespace freud::environment
