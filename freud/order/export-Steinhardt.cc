// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
// #include <nanobind/stl/list.h>      // NOLINT(misc-include-cleaner): used implicitly
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly

#include <utility>

#include "Steinhardt.h"
// #include "VectorMath.h"

namespace freud { namespace order {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap {

// void computeNematic(const std::shared_ptr<Nematic>& self,
// const nb_array<float, nanobind::shape<-1,3>>& orientations)
// {
// unsigned int const num_orientations = orientations.shape(0);
// auto* orientations_data = reinterpret_cast<vec3<float>*>(orientations.data());

// self->compute(orientations_data, num_orientations);
// }

}; // namespace wrap

namespace detail {

void export_Steinhardt(nanobind::module_& m)
{
    nanobind::class_<Steinhardt>(m, "Steinhardt")
        .def(nanobind::init<std::vector<unsigned int>, bool, bool, bool, bool>())
        .def("compute", &Steinhardt::compute, nanobind::arg("nlist").none(), nanobind::arg("points"), nanobind::arg("qargs"))
        .def("isAverage", &Steinhardt::isAverage)
        .def("isWl", &Steinhardt::isWl)
        .def("isWeighted", &Steinhardt::isWeighted)
        .def("isWlNormalized", &Steinhardt::isWlNormalized)
        .def("getL", &Steinhardt::getL)
        .def("getParticleOrder", &Steinhardt::getParticleOrder)
        .def("getQlm", &Steinhardt::getQlm)
        ;
}

} // namespace detail

}; }; // namespace freud::order
