// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly
#include <utility>

#include "Cubatic.h"
#include "VectorMath.h"

namespace freud { namespace order {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap {

void computeCubatic(const std::shared_ptr<Cubatic>& self,
                    const nb_array<float, nanobind::shape<-1, 4>>& orientations)
{
    unsigned int const num_orientations = orientations.shape(0);
    auto* orientations_data = reinterpret_cast<quat<float>*>(orientations.data());

    self->compute(orientations_data, num_orientations);
}

nanobind::tuple getCubaticOrientation(const std::shared_ptr<Cubatic>& self)
{
    quat<float> q = self->getCubaticOrientation();
    return nanobind::make_tuple(q.s, q.v.x, q.v.y, q.v.z);
}
}; // namespace wrap

namespace detail {

void export_Cubatic(nanobind::module_& m)
{
    nanobind::class_<Cubatic>(m, "Cubatic")
        .def(nanobind::init<float, float, float, unsigned int, unsigned int>())
        .def("compute", &wrap::computeCubatic, nanobind::arg("orientations"))
        .def("getTInitial", &Cubatic::getTInitial)
        .def("getTFinal", &Cubatic::getTFinal)
        .def("getScale", &Cubatic::getScale)
        .def("getNReplicates", &Cubatic::getNReplicates)
        .def("getSeed", &Cubatic::getSeed)
        .def("getCubaticOrderParameter", &Cubatic::getCubaticOrderParameter)
        .def("getCubaticOrientation", &wrap::getCubaticOrientation)
        .def("getParticleOrderParameter", &Cubatic::getParticleOrderParameter)
        .def("getGlobalTensor", &Cubatic::getGlobalTensor)
        .def("getCubaticTensor", &Cubatic::getCubaticTensor);
}

} // namespace detail

}; }; // namespace freud::order
