// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <cstdint>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly

#include "ClusterProperties.h"
#include "NeighborQuery.h"

namespace nb = nanobind;

namespace freud { namespace cluster {
template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace {
void compute(const std::shared_ptr<ClusterProperties>& self,
             const std::shared_ptr<locality::NeighborQuery>& nq,
             const nb_array<const uint32_t, nanobind::shape<-1>>& cluster_idx,
             const nb_array<const float, nanobind::shape<-1>>& masses)
{
    const auto* masses_data = reinterpret_cast<const float*>(masses.data());
    const auto* cluster_idx_data = reinterpret_cast<const uint32_t*>(cluster_idx.data());
    self->compute(nq, cluster_idx_data, masses_data);
}
}; // end anonymous namespace

namespace detail {
// NOLINTNEXTLINE(misc-use-internal-linkage)
void export_ClusterProperties(nb::module_& module)
{
    nanobind::class_<ClusterProperties>(module, "ClusterProperties")
        .def(nb::init<>())
        .def("compute", &compute, nanobind::arg("nq"), nanobind::arg("cluster_idx"),
             nanobind::arg("masses_data").none())
        .def("getClusterCenters", &ClusterProperties::getClusterCenters)
        .def("getClusterCentersOfMass", &ClusterProperties::getClusterCentersOfMass)
        .def("getClusterMomentsOfInertia", &ClusterProperties::getClusterMomentsOfInertia)
        .def("getClusterGyrations", &ClusterProperties::getClusterGyrations)
        .def("getClusterSizes", &ClusterProperties::getClusterSizes)
        .def("getClusterMasses", &ClusterProperties::getClusterMasses);
};

}; // end namespace detail

}; }; // namespace freud::cluster
