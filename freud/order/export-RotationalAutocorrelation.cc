// Copyright (c) 2010-2026 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly

#include "RotationalAutocorrelation.h"
#include "VectorMath.h"

namespace freud { namespace order {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap {

void computeRotationalAutocorrelation(const std::shared_ptr<RotationalAutocorrelation>& self,
                                      const nb_array<const float, nanobind::shape<-1, 4>>& ref_orientations,
                                      const nb_array<const float, nanobind::shape<-1, 4>>& orientations)
{
    unsigned int const num_orientations = orientations.shape(0);
    const auto* ref_orientations_data = reinterpret_cast<const quat<float>*>(ref_orientations.data());
    const auto* orientations_data = reinterpret_cast<const quat<float>*>(orientations.data());

    self->compute(ref_orientations_data, orientations_data, num_orientations);
}

}; // namespace wrap

namespace detail {

void export_RotationalAutocorrelation(nanobind::module_& m)
{
    nanobind::class_<RotationalAutocorrelation>(m, "RotationalAutocorrelation")
        .def(nanobind::init<unsigned int>())
        .def("compute", &wrap::computeRotationalAutocorrelation, nanobind::arg("ref_orientations"),
             nanobind::arg("orientations"))
        .def("getL", &RotationalAutocorrelation::getL)
        .def("getRotationalAutocorrelation", &RotationalAutocorrelation::getRotationalAutocorrelation)
        .def("getRAArray", &RotationalAutocorrelation::getRAArray);
}

} // namespace detail

}; }; // namespace freud::order
