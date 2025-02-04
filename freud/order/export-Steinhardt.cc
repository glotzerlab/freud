// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly
#include <nanobind/stl/vector.h>

#include "Steinhardt.h"

namespace freud { namespace order {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace detail {

void export_Steinhardt(nanobind::module_& m)
{
    nanobind::class_<Steinhardt>(m, "Steinhardt")
        .def(nanobind::init<std::vector<unsigned int>, bool, bool, bool, bool>())
        .def("compute", &Steinhardt::compute, nanobind::arg("nlist").none(), nanobind::arg("points"),
             nanobind::arg("qargs"))
        .def("isAverage", &Steinhardt::isAverage)
        .def("isWl", &Steinhardt::isWl)
        .def("isWeighted", &Steinhardt::isWeighted)
        .def("isWlNormalized", &Steinhardt::isWlNormalized)
        .def("getL", &Steinhardt::getL)
        .def("getOrder", &Steinhardt::getOrder)
        .def("getParticleOrder", &Steinhardt::getParticleOrder)
        .def("getQlm", &Steinhardt::getQlm)
        .def("getQl", &Steinhardt::getQl);
}

} // namespace detail

}; }; // namespace freud::order
