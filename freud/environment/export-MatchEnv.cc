// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <map>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>   // NOLINT(misc-include-cleaner): used implicitly
#include <nanobind/stl/map.h>        // NOLINT(misc-include-cleaner): used implicitly
#include <nanobind/stl/pair.h>       // NOLINT(misc-include-cleaner): used implicitly
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly
#include <nanobind/stl/vector.h>     // NOLINT(misc-include-cleaner): used implicitly
#include <utility>
#include <vector>

#include "Box.h"
#include "MatchEnv.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "VectorMath.h"

namespace nb = nanobind;

namespace freud { namespace environment {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap {
void compute_env_motif_match(const std::shared_ptr<EnvironmentMotifMatch>& env_motif_match,
                             const std::shared_ptr<locality::NeighborQuery>& nq,
                             const std::shared_ptr<locality::NeighborList>& nlist,
                             const locality::QueryArgs& qargs,
                             const nb_array<const float, nanobind::shape<-1, 3>>& motif,
                             const unsigned int motif_size, const float threshold, const bool registration)
{
    const auto* motif_data = reinterpret_cast<const vec3<float>*>(motif.data());
    env_motif_match->compute(nq, nlist, qargs, motif_data, motif_size, threshold, registration);
}

void compute_env_rmsd_min(const std::shared_ptr<EnvironmentRMSDMinimizer>& env_rmsd_min,
                          const std::shared_ptr<locality::NeighborQuery>& nq,
                          const std::shared_ptr<locality::NeighborList>& nlist,
                          const locality::QueryArgs& qargs,
                          const nb_array<const float, nanobind::shape<-1, 3>>& motif, const unsigned int motif_size,
                          const bool registration)
{
    const auto* motif_data = reinterpret_cast<const vec3<float>*>(motif.data());
    env_rmsd_min->compute(nq, nlist, qargs, motif_data, motif_size, registration);
}

std::pair<float, std::map<unsigned int, unsigned int>>
compute_minimize_RMSD(const box::Box& box, const nb_array<const float, nanobind::shape<-1, 3>>& refPoints1,
                      nb_array<float, nanobind::shape<-1, 3>>& refPoints2, unsigned int numRef,
                      float min_rmsd, bool registration)
{
    float min_rmsd_modified = min_rmsd;
    const auto* refPoints1_data = reinterpret_cast<const vec3<float>*>(refPoints1.data());
    auto* refPoints2_data = reinterpret_cast<vec3<float>*>(refPoints2.data());
    return std::make_pair(
        min_rmsd_modified,
        minimizeRMSD(box, refPoints1_data, refPoints2_data, numRef, min_rmsd_modified, registration));
}

std::map<unsigned int, unsigned int>
compute_is_similar(const box::Box& box, const nb_array<const float, nanobind::shape<-1, 3>>& refPoints1,
                   nb_array<float, nanobind::shape<-1, 3>>& refPoints2, unsigned int numRef,
                   float threshold_sq, bool registration)
{
    const auto* refPoints1_data = reinterpret_cast<const vec3<float>*>(refPoints1.data());
    auto* refPoints2_data = reinterpret_cast<vec3<float>*>(refPoints2.data());
    return isSimilar(box, refPoints1_data, refPoints2_data, numRef, threshold_sq, registration);
}

nb::list convertVectorsToLists(const std::vector<std::vector<vec3<float>>>& vecs)
{
    nb::list vecs_python;
    for (const auto& vec : vecs)
    {
        nb::list vec_python;
        for (const auto& v : vec)
        {
            nb::list v_python;
            v_python.append(v.x);
            v_python.append(v.y);
            v_python.append(v.z);
            vec_python.append(v_python);
        }
        vecs_python.append(vec_python);
    }
    return vecs_python;
}

nb::object getClusterEnv(const std::shared_ptr<EnvironmentCluster>& env_cls)
{
    return convertVectorsToLists(env_cls->getClusterEnvironments());
}

nb::object getPointEnv(const std::shared_ptr<EnvironmentCluster>& env_cls)
{
    return convertVectorsToLists(env_cls->getPointEnvironments());
}

nb::object getPointEnvmm(const std::shared_ptr<EnvironmentMotifMatch>& env_cls)
{
    return convertVectorsToLists(env_cls->getPointEnvironments());
}

}; // namespace wrap

namespace detail {

void export_MatchEnv(nb::module_& module)
{
    module.def("minimizeRMSD", &wrap::compute_minimize_RMSD);

    module.def("isSimilar", &wrap::compute_is_similar);

    nb::class_<MatchEnv>(module, "MatchEnv")
        .def(nb::init<>())
        .def("getPointEnvironments", &MatchEnv::getPointEnvironments);

    nb::class_<EnvironmentCluster>(module, "EnvironmentCluster")
        .def(nb::init<>())
        .def("compute", &EnvironmentCluster::compute, nb::arg("nq"), nb::arg("nlist").none(),
             nb::arg("qargs"), nb::arg("env_nlist").none(), nb::arg("env_qargs"), nb::arg("threshold"),
             nb::arg("registration"))
        .def("getClusterEnvironments", &wrap::getClusterEnv)
        .def("getPointEnvironments", &wrap::getPointEnv)
        .def("getClusters", &EnvironmentCluster::getClusters)
        .def("getNumClusters", &EnvironmentCluster::getNumClusters);

    nb::class_<EnvironmentMotifMatch>(module, "EnvironmentMotifMatch")
        .def(nb::init<>())
        .def("compute", &wrap::compute_env_motif_match, nb::arg("nq"), nb::arg("nlist").none(),
             nb::arg("qargs"), nb::arg("motif"), nb::arg("motif_size"), nb::arg("threshold"),
             nb::arg("registration"))
        .def("getPointEnvironments", &wrap::getPointEnvmm)
        .def("getMatches", &EnvironmentMotifMatch::getMatches);

    nb::class_<EnvironmentRMSDMinimizer>(module, "EnvironmentRMSDMinimizer")
        .def(nb::init<>())
        .def("compute", &wrap::compute_env_rmsd_min, nb::arg("nq"), nb::arg("nlist").none(), nb::arg("qargs"),
             nb::arg("motif"), nb::arg("motif_size"), nb::arg("registration"))
        .def("getRMSDs", &EnvironmentRMSDMinimizer::getRMSDs);
}

}; // namespace detail
}; }; // namespace freud::environment
