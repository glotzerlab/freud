// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly
#include <nanobind/stl/vector.h> 

#include <utility>

#include "ClusterProperties.h"

namespace nb = nanobind;

namespace freud { namespace cluster {
template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap {
        void compute(std::shared_ptr<ClusterProperties> & self, std::shared_ptr<locality::NeighborQuery> nq, nb_array<unsigned int, nanobind::shape<-1>> cluster_idx,
                 nb_array<float, nanobind::shape<-1>> masses)
{
        float* masses_data = reinterpret_cast<float*>(masses.data());
        unsigned int* cluster_idx_data = reinterpret_cast<unsigned int*>(cluster_idx.data());
        self->compute(nq, cluster_idx_data, masses_data);
}
}; //end namespace wrap

namespace detail {
    void export_ClusterProperties(nb::module_& module)
    {
        nanobind::class_<ClusterProperties>(module, "ClusterProperties")
            .def(nb::init<>())
            .def("compute", &wrap::compute, nanobind::arg("nq"), nanobind::arg("cluster_idx"), nanobind::arg("masses_data").none())
            .def("getClusterCenters", &ClusterProperties::getClusterCenters)
            .def("getClusterCentersOfMass", &ClusterProperties::getClusterCentersOfMass)
            .def("getClusterMomentsOfInertia", &ClusterProperties::getClusterMomentsOfInertia)
            .def("getClusterGyrations", &ClusterProperties::getClusterGyrations)
            .def("getClusterSizes", &ClusterProperties::getClusterSizes)
            .def("getClusterMasses", &ClusterProperties::getClusterMasses);
    };

}; //end namespace detail

}; };
//m.def("func", &func, "arg"_a.none());