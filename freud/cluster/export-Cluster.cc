// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly
#include <nanobind/stl/vector.h> 

#include <utility>

#include "Cluster.h"

namespace nb = nanobind;

namespace freud { namespace cluster {
template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap {
        void compute(std::shared_ptr<Cluster> self, std::shared_ptr<locality::NeighborQuery> nq, std::shared_ptr<locality::NeighborList> nlist,
                const locality::QueryArgs& qargs, nb_array<unsigned int, nanobind::shape<-1>> keys)
{
        unsigned int* keys_data = reinterpret_cast<unsigned int*>(keys.data());
        self->compute(nq, nlist, qargs, keys_data);
}
}; //end namespace wrap

namespace detail {
    void export_Cluster(nb::module_& module)
    {
        nanobind::class_<Cluster>(module, "Cluster")
            .def(nb::init<>())
            .def("compute", &wrap::compute, nanobind::arg("nq"), nanobind::arg("nlist"), nanobind::arg("qargs"), nanobind::arg("keys").none())
            .def("getNumClusters", &Cluster::getNumClusters)
            .def("getClusterIdx", &Cluster::getClusterIdx)
            .def("getClusterKeys", &Cluster::getClusterKeys);
    };

}; //end namespace detail

}; };
//m.def("func", &func, "arg"_a.none());