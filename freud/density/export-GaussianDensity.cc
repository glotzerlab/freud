// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.


#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly

#include <utility>

#include "GaussianDensity.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "VectorMath.h"


namespace freud { namespace density {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap {

void computeDensity(const std::shared_ptr<GaussianDensity>& self, const std::shared_ptr<locality::NeighborQuery>& nq,
                    const nb_array<float, nanobind::shape<-1>>& values)
{
    const float* values_data = values.is_valid() ? reinterpret_cast<float*>(values.data()) : nullptr;
    self->compute(nq.get(), values_data);
}

}; // namespace wrap

namespace detail {

void export_GaussianDensity(nanobind::module_& m)
{
    nanobind::class_<GaussianDensity>(m, "GaussianDensity")
        .def(nanobind::init<vec3<unsigned int>, float, float>())
        .def("compute", &wrap::computeDensity, nanobind::arg("nq"), nanobind::arg("values") = nullptr)
        .def("getDensity", &GaussianDensity::getDensity)
        .def("getBox", &GaussianDensity::getBox)
        .def("getWidth", &GaussianDensity::getWidth)
        .def("getSigma", &GaussianDensity::getSigma)
        .def("getRMax", &GaussianDensity::getRMax);
}

} // namespace detail

}; }; // end namespace freud::density
