// Copyright (c) 2010-2024 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly
#include <nanobind/stl/function.h> // NOLINT(misc-include-cleaner): used implicitly
#include <nanobind/stl/map.h> 
#include <nanobind/stl/vector.h>     // NOLINT(misc-include-cleaner): used implicitly

#include "MatchEnv.h"
#include "Registration.h"


namespace nb = nanobind;

namespace freud { namespace environment {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap {
void compute_env_motif_match(const std::shared_ptr<EnvironmentMotifMatch>& env_motif_match, 
             std::shared_ptr<locality::NeighborQuery> nq,
             std::shared_ptr<locality::NeighborList> nlist,
             const locality::QueryArgs& qargs,
             const nb_array<float, nanobind::shape<-1, 3>>& motif,
             const unsigned int motif_size,
             const float threshold,
             const bool registration
)
{
    auto* motif_data = reinterpret_cast<vec3<float>*>(motif.data());
    env_motif_match->compute(nq, nlist, qargs, motif_data, motif_size, threshold, registration);
}

void compute_env_rmsd_min(const std::shared_ptr<EnvironmentRMSDMinimizer>& env_rmsd_min, 
             std::shared_ptr<locality::NeighborQuery> nq,
             std::shared_ptr<locality::NeighborList> nlist,
             const locality::QueryArgs& qargs,
             const nb_array<float, nanobind::shape<-1, 3>>& motif,
             const unsigned int motif_size,
             const float threshold,
             const bool registration
)
{
    auto* motif_data = reinterpret_cast<vec3<float>*>(motif.data());
    // TODO: where should threshold go?
    env_rmsd_min->compute(nq, nlist, qargs, motif_data, motif_size, registration);
}

};

namespace detail {

// 1. Define the function pointer type for the overload that returns a std::map<unsigned int, unsigned int>.
using MinimizeRMSD_Coords = std::map<unsigned int, unsigned int> (*)(
    const box::Box&,
    const vec3<float>*,
    vec3<float>*,
    unsigned int,
    float&, 
    bool);

using IsSimilar_Coords = std::map<unsigned int, unsigned int> (*)(
    const box::Box&,
    const vec3<float>*,
    vec3<float>*,
    unsigned int,
    float,
    bool);

void export_MatchEnv(nb::module_& module)
{
    module.def(
        "minimizeRMSD",
        (MinimizeRMSD_Coords) &minimizeRMSD,
        "Compute a map of matching indices between two sets of points. This overload also potentially\n"
        "modifies the second set of points in-place if registration=True.\n\n"
        "Args:\n"
        "    box (Box): Simulation box.\n"
        "    refPoints1 (array of vec3<float>): Points in the first environment.\n"
        "    refPoints2 (array of vec3<float>): Points in the second environment (modified if registration=True).\n"
        "    numRef (int): Number of points.\n"
        "    min_rmsd (float): Updated by reference to the final RMSD.\n"
        "    registration (bool): If True, perform a brute-force alignment.\n\n"
        "Returns:\n"
        "    dict(int->int): Index mapping from the first set of points to the second set.");
    
    module.def(
        "isSimilar",
        (IsSimilar_Coords) &isSimilar,
        "Check if two sets of points can be matched (i.e., are 'similar') within a given distance threshold.\n"
        "Potentially modifies the second set in-place if registration=True.\n\n"
        "Args:\n"
        "    box (Box): Simulation box.\n"
        "    refPoints1 (array of vec3<float>): Points in the first environment.\n"
        "    refPoints2 (array of vec3<float>): Points in the second environment.\n"
        "    numRef (int): Number of points.\n"
        "    threshold_sq (float): Square of the max distance allowed to consider points matching.\n"
        "    registration (bool): If True, attempt brute force alignment.\n\n"
        "Returns:\n"
        "    dict(int->int): Index mapping if the environments match, else an empty map."
    );

    nb::class_<MatchEnv>(module, "MatchEnv")
        .def(nb::init<>())
        .def("getPointEnvironments", &MatchEnv::getPointEnvironments);

    nb::class_<EnvironmentCluster>(module, "EnvironmentCluster")
        .def(nb::init<>())
        .def("compute", &EnvironmentCluster::compute)
        // .def("getClusters", &EnvironmentCluster::getClusterIdx) // TODO: should be there
        .def("getClusterEnvironments", &EnvironmentCluster::getClusterEnvironments)
        .def("getNumClusters", &EnvironmentCluster::getNumClusters);

    nb::class_<EnvironmentMotifMatch>(module, "EnvironmentMotifMatch")
        .def(nb::init<>())
        .def("compute", &wrap::compute_env_motif_match, nb::arg("nq"), nb::arg("nlist"), nb::arg("qargs"), nb::arg("motif"), nb::arg("motif_size"), nb::arg("threshold"), nb::arg("registration"))
        .def("getMatches", &EnvironmentMotifMatch::getMatches);

    nb::class_<EnvironmentRMSDMinimizer>(module, "EnvironmentRMSDMinimizer")
        .def(nb::init<>())
        .def("compute", &wrap::compute_env_rmsd_min, nb::arg("nq"), nb::arg("nlist"), nb::arg("qargs"), nb::arg("motif"), nb::arg("motif_size"), nb::arg("threshold"), nb::arg("registration"))
        .def("getRMSDs", &EnvironmentRMSDMinimizer::getRMSDs);

}

}; }; // namespace detail
}; // namespace freud::locality
