// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly
#include <nanobind/stl/vector.h>     // NOLINT(misc-include-cleaner): used implicitly

#include "AngularSeparation.h"


namespace nb = nanobind;

namespace freud { namespace environment {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap {
void compute(const std::shared_ptr<AngularSeparationGlobal> &self, 
        const nb_array<float, nanobind::shape<-1, 4>> &global_orientations,
             const nb_array<float, nanobind::shape<-1, 4>> & orientations,
             const nb_array<float, nanobind::shape<-1, 4>> & equiv_orientations)
    {
    unsigned int const n_global = global_orientations.shape(0);
    unsigned int const n_points = orientations.shape(0);
    unsigned int const n_equiv_orientations = equiv_orientations.shape(0);
    auto* global_orientations_data = reinterpret_cast<quat<float>*>(global_orientations.data());
    auto* orientations_data = reinterpret_cast<quat<float>*>(orientations.data());
    auto* equiv_orientations_data = reinterpret_cast<quat<float>*>(equiv_orientations.data());
    self->compute(global_orientations_data, n_global, orientations_data, n_points, equiv_orientations_data, n_equiv_orientations);
    }

};

namespace detail {

void export_AngularSeparationGlobal(nb::module_& module)
{
    nb::class_<AngularSeparationGlobal>(module, "AngularSeparationGlobal")
        .def(nb::init<>())
        .def("getAngles", &AngularSeparationGlobal::getAngles)
        .def("compute", &AngularSeparationGlobal::compute);
}

}; }; // namespace detail
}; // namespace freud::locality
