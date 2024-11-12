// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly
#include <nanobind/stl/vector.h>     // NOLINT(misc-include-cleaner): used implicitly

#include "MatchEnv.h"
#include "Registration.h"


namespace nb = nanobind;

namespace freud { namespace environment {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap {
void compute(const std::shared_ptr<MatchEnv>& match_env, 
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
    match_env->compute(nq, query_points_data, n_query_points, orientations_data, nlist, qargs, max_num_neighbors);
    }

};

namespace detail {

void export_MatchEnv(nb::module_& module)
{
    // export minimizeRMSD function
    // export isSimilar function
    // export MatchEnv class
        // export getPointEnvironments fn
    nb::class_<MatchEnv>(module, "MatchEnv")
        .def(nb::init<>)
        .def("getPointEnvironments", &MatchEnv::getPointEnvironments)
    // export EnvironmentCluster class 
        // export compute fn
        // export getClusterIdx fn
        // export getClusterEnvironments fn
        // export getNumClusters fn
    // export EnvironmentMotifMatch class
        // export compute fn
        // export getMatches fn
    // export EnvironmentRMSDMinimizer class
        // export compute fn
        // export getRMSDs fn

}

}; }; // namespace detail
}; // namespace freud::locality
