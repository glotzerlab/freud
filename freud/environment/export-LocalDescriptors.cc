// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly
#include <nanobind/stl/vector.h>     // NOLINT(misc-include-cleaner): used implicitly

#include "LocalDescriptors.h"


namespace nb = nanobind;

namespace freud { namespace environment {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap {
void compute(const std::shared_ptr<LocalDescriptors>& local_descriptors, 
             std::shared_ptr<locality::NeighborQuery> nq,
             const nb_array<float, nanobind::shape<-1, 3>>& query_points,
             const unsigned int n_query_points,
             const nb_array<float, nanobind::shape<-1, 4>>& orientations,
             std::shared_ptr<locality::NeighborList> nlist,
             const locality::QueryArgs& qargs,
             const unsigned int max_num_neighbors
)
    {
    auto* query_points_data = reinterpret_cast<vec3<float>*>(query_points.data());
    auto* orientations_data = reinterpret_cast<quat<float>*>(orientations.data());
    local_descriptors->compute(nq, query_points_data, n_query_points, orientations_data, nlist, qargs, max_num_neighbors);
    }

};

namespace detail {

void export_LocalDescriptors(nb::module_& module)
{

    nb::enum_<LocalDescriptorOrientation>(module, "LocalDescriptorOrientation")
        .value("LocalNeighborhood", LocalDescriptorOrientation::LocalNeighborhood)
        .value("Global", LocalDescriptorOrientation::Global)
        .value("ParticleLocal", LocalDescriptorOrientation::ParticleLocal)
        .export_values();

    nb::class_<LocalDescriptors>(module, "LocalDescriptors")
        .def(nb::init<unsigned int, bool, LocalDescriptorOrientation>())
        .def("getNList", &LocalDescriptors::getNList)
        .def("getSph", &LocalDescriptors::getSph)
        .def("getNSphs", &LocalDescriptors::getNSphs)
        .def("getLMax", &LocalDescriptors::getLMax)
        .def("getNegativeM", &LocalDescriptors::getNegativeM)
        .def("getMode", &LocalDescriptors::getMode)
        .def("compute", &wrap::compute,nb::arg("nq"),  
             nb::arg("query_points"), nb::arg("n_query_points"),  nb::arg("orientations").none(),
             nb::arg("nlist").none(),
             nb::arg("qargs"), nb::arg("max_num_neighbors"));
}

}; }; // namespace detail
}; // namespace freud::locality
