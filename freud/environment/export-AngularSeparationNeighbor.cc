// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly
#include <nanobind/stl/vector.h>     // NOLINT(misc-include-cleaner): used implicitly
#include <vector>

#include "AngularSeparation.h"

namespace nb = nanobind;

namespace freud { namespace environment {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap {

void compute(
    std::shared_ptr<AngularSeparationNeighbor> &self,
    std::shared_ptr<locality::NeighborQuery> &nq,
    nb_array<float, nanobind::shape<-1, 4>> & orientations,
    nb_array<float, nanobind::shape<-1, 3>> & query_points,
    nb_array<float, nanobind::shape<-1, 4>> & query_orientations,
    nb_array<float, nanobind::shape<-1, 4>> & equiv_orientations,
    std::shared_ptr<locality::NeighborList> &nlist,
    const locality::QueryArgs qargs
)
    {
    auto* orientations_data = reinterpret_cast<quat<float>*>(orientations.data());
    auto* query_points_data = reinterpret_cast<vec3<float>*>(query_points.data());
    auto* query_orientations_data = reinterpret_cast<quat<float>*>(query_orientations.data());
    auto* equiv_orientations_data = reinterpret_cast<quat<float>*>(equiv_orientations.data());
    unsigned int n_equiv_orientations = equiv_orientations.shape(0);
    unsigned int n_query_points = query_orientations.shape(0);
    self->compute(nq, orientations_data, query_points_data, query_orientations_data, n_query_points, equiv_orientations_data, n_equiv_orientations, nlist, qargs);
    }
}

namespace detail {

void export_AngularSeparationNeighbor(nb::module_& module)
{
    nb::class_<AngularSeparationNeighbor>(module, "AngularSeparationNeighbor")
        .def("getNList", &AngularSeparationNeighbor::getNList)
        .def("getAngles", &AngularSeparationNeighbor::getAngles)
        .def("compute", &AngularSeparationNeighbor::compute);
}

}; // namespace detail

}; }; // namespace freud::locality
