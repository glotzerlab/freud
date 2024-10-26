// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly

#include <utility>

#include "ContinuousCoordination.h"

namespace freud { namespace order {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace detail {

void export_ContinuousCoordination(nanobind::module_& m)
{
    nanobind::class_<ContinuousCoordination>(m, "ContinuousCoordination")
        .def(nanobind::init<std::vector<float>, bool, bool>())
        .def("compute", &ContinuousCoordination::compute, nanobind::arg("voronoi"))
        .def("getCoordination", &ContinuousCoordination::getCoordination)
        .def("getPowers", &ContinuousCoordination::getPowers)
        .def("getComputeLog", &ContinuousCoordination::getComputeLog)
        .def("getComputeExp", &ContinuousCoordination::getComputeExp)
        .def("getNumberOfCoordinations", &ContinuousCoordination::getNumberOfCoordinations)
        ;
}

} // namespace detail

}; }; // namespace freud::order
