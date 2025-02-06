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
std::shared_ptr<GaussianDensity> make_gaussian_density(unsigned int width_x, unsigned int width_y,
                                                       unsigned int width_z, float r_max, float sigma)
{
    return std::make_shared<GaussianDensity>(vec3<unsigned int>(width_x, width_y, width_z), r_max, sigma);
}

nanobind::tuple get_width(std::shared_ptr<GaussianDensity> self)
{
    auto width = self->getWidth();
    return nanobind::make_tuple(width.x, width.y, width.z);
}

void computeDensity(const std::shared_ptr<GaussianDensity>& self,
                    const std::shared_ptr<locality::NeighborQuery>& nq,
                    const nb_array<float, nanobind::shape<-1>>& values)
{
    const auto* values_data = values.is_valid() ? reinterpret_cast<const float*>(values.data()) : nullptr;
    self->compute(nq.get(), values_data);
}

}; // namespace wrap

namespace detail {

void export_GaussianDensity(nanobind::module_& m)
{
    m.def("make_gaussian_density", &wrap::make_gaussian_density);
    nanobind::class_<GaussianDensity>(m, "GaussianDensity")
        .def(nanobind::init<vec3<unsigned int>, float, float>())
        .def("compute", &wrap::computeDensity, nanobind::arg("nq"), nanobind::arg("values") = nullptr)
        .def_prop_ro("density", &GaussianDensity::getDensity)
        .def("getBox", &GaussianDensity::getBox)
        .def("getWidth", &wrap::get_width)
        .def("getSigma", &GaussianDensity::getSigma)
        .def("getRMax", &GaussianDensity::getRMax);
}

} // namespace detail

}; }; // end namespace freud::density
