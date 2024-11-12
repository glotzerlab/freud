// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly
#include <nanobind/stl/vector.h>     // NOLINT(misc-include-cleaner): used implicitly

#include "Cluster.h"
#include "NeighborList.h"
#include "NeighborQuery.h"

namespace nb = nanobind;

namespace freud { namespace cluster {
template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace {
void compute(const std::shared_ptr<Cluster>& self, const std::shared_ptr<locality::NeighborQuery>& nq,
             std::shared_ptr<locality::NeighborList>& nlist, const locality::QueryArgs& qargs,
             const nb_array<const unsigned int, nanobind::shape<-1>>& keys)
{
    const auto* keys_data = reinterpret_cast<const unsigned int*>(keys.data());
    self->compute(nq, nlist, qargs, keys_data);
}
}; // end anonymous namespace

namespace detail {
// NOLINTNEXTLINE(misc-use-internal-linkage)
void export_Cluster(nb::module_& module)
{
    nanobind::class_<Cluster>(module, "Cluster")
        .def(nb::init<>())
        .def("compute", &compute, nanobind::arg("nq"), nanobind::arg("nlist").none(), nanobind::arg("qargs"),
             nanobind::arg("keys").none())
        .def("getNumClusters", &Cluster::getNumClusters)
        .def("getClusterIdx", &Cluster::getClusterIdx)
        .def("getClusterKeys", &Cluster::getClusterKeys);
};

}; // end namespace detail

}; }; // namespace freud::cluster
